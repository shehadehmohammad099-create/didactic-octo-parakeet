#!/usr/bin/env python3
"""
Two-way Telegram bot that replies in your texting style using Claude.
Runs on Render (free) using webhook mode.

Features:
- Replies in your texting style (short, lowercase, "u"/"ur", 😭, etc.)
- Big supportive energy: asks questions, checks in, actually engaged
- Knows your personal context (PERSONAL_CONTEXT env var)
- Can see images you send and react to them
- Sends replies as 1-3 separate messages, like real texting
- PERSISTENT MEMORY via Postgres -- remembers across restarts/sleeps

Environment variables (set in Render dashboard):
    ANTHROPIC_API_KEY   - your Claude API key
    TELEGRAM_TOKEN      - the token @BotFather gives you
    WEBHOOK_SECRET      - any random string you make up
    ALLOWED_CHAT_ID     - your numeric Telegram chat id (bot only replies to you)
    PERSONAL_CONTEXT    - your context blurb (kept out of the code/GitHub)
    DATABASE_URL        - Postgres connection string (Render sets this
                          automatically when you attach a Render Postgres DB)
"""

import base64
import os
import time

import anthropic
import psycopg2
import psycopg2.extras
import requests
from flask import Flask, request

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
ALLOWED_CHAT_ID = os.environ.get("ALLOWED_CHAT_ID", "").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

PERSONAL_CONTEXT = os.environ.get(
    "PERSONAL_CONTEXT",
    "You don't have specific context about this person yet -- just be a warm, supportive friend.",
)

STYLE_PROMPT = f"""You are texting as a close, supportive friend. You know this person well.

Here's what you know about them:
{PERSONAL_CONTEXT}

TEXTING STYLE -- match this exactly:
- Keep it SHORT: mostly 3-6 words per message. Rarely more than one short sentence per bubble.
- Mostly lowercase, especially at the start. Skip ending punctuation (no periods).
- Use "u" and "ur" instead of "you"/"your".
- Occasionally use abbreviations naturally: ig, abt, js, tmrw, idk, nvm.
- Rarely use "!" -- but when you do, double it up ("!!").
- Occasional ALL CAPS for a sudden spike of excitement/shock/exasperation.
- 😭 is the go-to emoji, used sparingly.
- No corporate/assistant tone, no "I'm an AI" disclaimers.

PERSONALITY -- big supportive energy:
- Be genuinely warm, curious, and engaged. Ask follow-up questions often.
- Check in on things they mentioned before. Remember what's going on in their life.
- Hype them up, validate their feelings, be in their corner.
- Still text like a real friend though -- supportive doesn't mean long-winded or preachy.
  Keep it casual and real, not therapy-speak.

MESSAGE FORMAT -- text like a real person, in bursts:
- Split your reply into 1 to 3 SHORT separate messages, the way people actually text.
- Put each separate message on its own line, with "|||" between them.
- Example: "wait fr??|||thats actually so exciting|||how u feeling abt it"
- Sometimes just ONE message is right. Don't force three.

Output ONLY the message text (with ||| between bubbles if multiple). Nothing else."""


# ---------------------------------------------------------------------------
# Persistence layer (Postgres)
# ---------------------------------------------------------------------------
def db_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create the messages table if it doesn't exist. Called once at startup."""
    if not DATABASE_URL:
        print("WARNING: no DATABASE_URL set -- running with NO persistent memory.")
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id       SERIAL PRIMARY KEY,
                chat_id  TEXT NOT NULL,
                role     TEXT NOT NULL,      -- 'user' or 'assistant'
                content  TEXT NOT NULL,      -- stored as plain text
                ts       TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages (chat_id, id)")
        conn.commit()
    print("DB ready.")


def load_history(chat_id: str, limit: int = 20):
    """Return the last `limit` turns for this chat as a messages list."""
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
            (chat_id, limit),
        )
        rows = cur.fetchall()
    # rows are newest-first; reverse to chronological
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_message(chat_id: str, role: str, content: str):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s)",
            (chat_id, role, content),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Claude + Telegram
# ---------------------------------------------------------------------------
def build_user_content(text: str, image_b64: str = None, media_type: str = None):
    if image_b64:
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            },
            {"type": "text", "text": text if text else "[they sent this photo]"},
        ]
    return text


def generate_reply(chat_id: str, user_content, text_for_storage: str) -> list:
    # Load prior turns from the DB, then append this new one for the API call.
    history = load_history(chat_id, limit=20)
    messages = history + [{"role": "user", "content": user_content}]

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=STYLE_PROMPT,
        messages=messages[-20:],
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip().strip('"')

    # Persist. For images we store a short text placeholder (can't store the
    # whole image in history cheaply), so future context still makes sense.
    save_message(chat_id, "user", text_for_storage)
    save_message(chat_id, "assistant", reply)

    bubbles = [b.strip() for b in reply.split("|||") if b.strip()]
    return bubbles or [reply]


def send_message(chat_id, text: str) -> None:
    r = requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    print("Send status:", r.status_code, r.text)


def send_typing(chat_id) -> None:
    requests.post(
        f"{TELEGRAM_API}/sendChatAction",
        json={"chat_id": chat_id, "action": "typing"},
        timeout=10,
    )


def download_telegram_image(file_id: str):
    r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=20)
    file_path = r.json().get("result", {}).get("file_path")
    if not file_path:
        return None, None
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    img = requests.get(file_url, timeout=20)
    b64 = base64.b64encode(img.content).decode("utf-8")
    media_type = "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return b64, media_type


@app.route("/", methods=["GET"])
def health():
    return "ok", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        return "forbidden", 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")
    if not message:
        return "ok", 200

    chat_id = message["chat"]["id"]

    if ALLOWED_CHAT_ID and str(chat_id) != ALLOWED_CHAT_ID:
        print(f"Ignoring message from chat {chat_id} (not allowed)")
        return "ok", 200

    text = message.get("text", "")
    caption = message.get("caption", "")
    photos = message.get("photo")

    image_b64 = media_type = None
    if photos:
        file_id = photos[-1]["file_id"]
        image_b64, media_type = download_telegram_image(file_id)
        text = caption

    if not text and not image_b64:
        return "ok", 200

    print(f"Incoming from {chat_id}: text={text!r} image={'yes' if image_b64 else 'no'}")
    send_typing(chat_id)

    user_content = build_user_content(text, image_b64, media_type)
    stored_text = text if text else "[sent a photo]"
    bubbles = generate_reply(str(chat_id), user_content, stored_text)

    for i, bubble in enumerate(bubbles):
        if i > 0:
            send_typing(chat_id)
            time.sleep(min(1.5, 0.4 + len(bubble) * 0.03))
        send_message(chat_id, bubble)

    return "ok", 200


# Initialize the DB table when the app starts (works under gunicorn too).
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
