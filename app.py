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
import random
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()  # for Whisper voice transcription

# Proactive check-in config
CHECKIN_SECRET = os.environ.get("CHECKIN_SECRET", "").strip()  # protects the /checkin endpoint
CHECKIN_PROBABILITY = float(os.environ.get("CHECKIN_PROBABILITY", "0.4"))  # chance per ping
CHECKIN_MIN_GAP_HOURS = float(os.environ.get("CHECKIN_MIN_GAP_HOURS", "5"))  # min hours between check-ins

# Local timezone so the bot knows the current date/time in your world.
# Uses an IANA name (e.g. "Europe/London", "America/New_York") which handles
# daylight saving automatically.
TIMEZONE = os.environ.get("TIMEZONE", "Europe/London")
try:
    LOCAL_TZ = ZoneInfo(TIMEZONE)
except Exception:
    print(f"Unknown TIMEZONE {TIMEZONE!r}, falling back to UTC")
    LOCAL_TZ = timezone.utc


def now_local_str() -> str:
    """Human-readable current local date/time, e.g. 'Monday, 13 July 2026, 3:42pm'."""
    now = datetime.now(LOCAL_TZ)
    # %-I / %-M are platform-specific; build the time part manually to be safe.
    hour12 = now.strftime("%I").lstrip("0") or "12"
    ampm = now.strftime("%p").lower()
    return f"{now.strftime('%A, %d %B %Y')}, {hour12}:{now.strftime('%M')}{ampm}"

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

PERSONAL_CONTEXT = os.environ.get(
    "PERSONAL_CONTEXT",
    "You don't have specific context about this person yet -- just be a warm, supportive friend.",
)

# The style/personality prompt is loaded from an env var so you can tweak the
# bot's personality in the Render dashboard without editing code or redeploying.
# Use the literal placeholder {PERSONAL_CONTEXT} anywhere in your prompt and the
# code will substitute your PERSONAL_CONTEXT value into it at runtime.
DEFAULT_STYLE_PROMPT = """You are texting as a close, supportive friend. You know this person well.

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

PERSONALITY -- warm, but with range:
- Default warmth: genuinely engaged, curious, in their corner. Ask follow-up questions.
- Be a SOUNDING BOARD, not just a cheerleader. When they float an idea, actually
  engage with it -- build on it, poke at it gently, offer a real opinion or advice.
- CONNECT THREADS: link what they say now to things they mentioned before. Notice
  patterns ("this is the second time uni's stressed u out this week" / "didnt u say
  u wanted to try that").
- Give actual advice when it'd help, not just validation. You can disagree or offer
  a different angle -- a good friend does.
- VARY YOUR ENERGY, read the moment: match hype with hype, be silly and playful when
  it's light, but get thoughtful and grounded when they're working through something
  real. Don't be relentlessly bubbly regardless of context.
- Still text like a real friend -- casual, short, never preachy or therapy-speak.

TIME AWARENESS:
- You'll be told the current date and time. USE IT. Reason about whether things are
  upcoming or already happened. If they say "im going fencing" that's the FUTURE --
  wish them luck, don't ask how it went. Only ask how something went once it's
  plausibly over. If they say "i cant wait" ask what for, don't assume.
- You can naturally reference time of day, day of week, etc when relevant.

MESSAGE FORMAT -- text like a real person, in bursts:
- Split your reply into 1 to 3 SHORT separate messages, the way people actually text.
- Put each separate message on its own line, with "|||" between them.
- Example: "wait fr??|||thats actually so exciting|||how u feeling abt it"
- Sometimes just ONE message is right. Don't force three.

Output ONLY the message text (with ||| between bubbles if multiple). Nothing else."""

# Use the env var if set, otherwise fall back to the default above.
_raw_style_prompt = os.environ.get("STYLE_PROMPT", DEFAULT_STYLE_PROMPT)

# Substitute PERSONAL_CONTEXT into the prompt. .replace (not .format) is used
# so that other curly braces in the prompt won't cause errors.
STYLE_PROMPT = _raw_style_prompt.replace("{PERSONAL_CONTEXT}", PERSONAL_CONTEXT)


# ---------------------------------------------------------------------------
# Persistence layer (Postgres)
# ---------------------------------------------------------------------------
def db_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create tables if they don't exist. Called once at startup."""
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

        # Long-term durable facts about the person (one row per fact).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id       SERIAL PRIMARY KEY,
                chat_id  TEXT NOT NULL,
                fact     TEXT NOT NULL,
                ts       TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_chat ON facts (chat_id)")

        # Per-chat rolling summary + a marker of how far it has summarized.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_state (
                chat_id            TEXT PRIMARY KEY,
                summary            TEXT DEFAULT '',
                summarized_upto_id INTEGER DEFAULT 0,
                last_checkin       TIMESTAMPTZ
            )
            """
        )
        # In case the table already existed without the column (older deploy):
        cur.execute("ALTER TABLE chat_state ADD COLUMN IF NOT EXISTS last_checkin TIMESTAMPTZ")
        conn.commit()
    print("DB ready.")


def load_recent_history(chat_id: str, limit: int = 15):
    """Return the last `limit` verbatim turns as a messages list.

    User turns get a light [time] prefix so the model can reason about when
    things were said (e.g. distinguishing "going fencing" said an hour ago from
    days ago). Assistant turns are left clean.
    """
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT role, content, ts FROM messages WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
            (chat_id, limit),
        )
        rows = cur.fetchall()

    out = []
    for role, content, ts in reversed(rows):
        if role == "user" and ts is not None:
            local = ts.astimezone(LOCAL_TZ)
            hour12 = local.strftime("%I").lstrip("0") or "12"
            stamp = f"{local.strftime('%a')} {hour12}:{local.strftime('%M')}{local.strftime('%p').lower()}"
            out.append({"role": role, "content": f"[{stamp}] {content}"})
        else:
            out.append({"role": role, "content": content})
    return out


def save_message(chat_id: str, role: str, content: str):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s)",
            (chat_id, role, content),
        )
        conn.commit()


# ---- Long-term facts ----
def load_facts(chat_id: str) -> list:
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT fact FROM facts WHERE chat_id = %s ORDER BY id", (chat_id,))
        return [r[0] for r in cur.fetchall()]


def add_facts(chat_id: str, new_facts: list):
    if not DATABASE_URL or not new_facts:
        return
    with db_conn() as conn, conn.cursor() as cur:
        for f in new_facts:
            cur.execute("INSERT INTO facts (chat_id, fact) VALUES (%s, %s)", (chat_id, f))
        conn.commit()


# ---- Rolling summary ----
def load_state(chat_id: str):
    """Return (summary, summarized_upto_id)."""
    if not DATABASE_URL:
        return "", 0
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT summary, summarized_upto_id FROM chat_state WHERE chat_id = %s",
            (chat_id,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else ("", 0)


def save_state(chat_id: str, summary: str, upto_id: int):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_state (chat_id, summary, summarized_upto_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (chat_id) DO UPDATE
              SET summary = EXCLUDED.summary,
                  summarized_upto_id = EXCLUDED.summarized_upto_id
            """,
            (chat_id, summary, upto_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Claude + Telegram
# ---------------------------------------------------------------------------
RECENT_TURNS = 15          # verbatim turns kept in the live prompt
SUMMARIZE_AFTER = 25       # once unsummarized older turns exceed this, fold them in


def extract_and_store_facts(chat_id: str, user_text: str, bot_reply: str, existing_facts: list):
    """Lightweight background call: pull any NEW durable facts from this exchange."""
    try:
        existing = "\n".join(f"- {f}" for f in existing_facts) or "(none yet)"
        prompt = (
            "From the exchange below, extract any NEW durable facts worth "
            "remembering long-term about the user (their life, relationships, "
            "job, plans, preferences, ongoing situations). Only NEW facts not "
            "already known. Ignore fleeting small talk. Return each fact on its "
            "own line, no bullets. If there are none, return exactly NONE.\n\n"
            f"Already known:\n{existing}\n\n"
            f"User: {user_text}\nFriend: {bot_reply}"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(b.text for b in resp.content if b.type == "text").strip()
        if out.upper() != "NONE":
            new_facts = [line.strip("-• ").strip() for line in out.splitlines() if line.strip()]
            new_facts = [f for f in new_facts if f and f.upper() != "NONE"]
            add_facts(chat_id, new_facts)
    except Exception as e:
        print("Fact extraction failed (non-fatal):", e)


def maybe_update_summary(chat_id: str):
    """If enough new older messages have piled up, fold them into the rolling summary."""
    if not DATABASE_URL:
        return
    try:
        summary, upto_id = load_state(chat_id)
        with db_conn() as conn, conn.cursor() as cur:
            # grab messages newer than what's summarized, EXCEPT the most recent
            # RECENT_TURNS (those stay verbatim in the live prompt)
            cur.execute(
                "SELECT id FROM messages WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
                (chat_id, RECENT_TURNS),
            )
            recent_ids = [r[0] for r in cur.fetchall()]
            floor_id = min(recent_ids) if recent_ids else None

            if floor_id is None:
                return
            cur.execute(
                """
                SELECT id, role, content FROM messages
                WHERE chat_id = %s AND id > %s AND id < %s
                ORDER BY id
                """,
                (chat_id, upto_id, floor_id),
            )
            to_fold = cur.fetchall()

        if len(to_fold) < SUMMARIZE_AFTER:
            return  # not enough new backlog yet

        transcript = "\n".join(f"{role}: {content}" for _id, role, content in to_fold)
        prompt = (
            "Update the running summary of a friendship chat. Keep it a concise "
            "paragraph capturing the important ongoing threads, events, and mood "
            "-- not a play-by-play. Merge the new messages into the existing "
            "summary.\n\n"
            f"Existing summary:\n{summary or '(empty)'}\n\n"
            f"New messages to fold in:\n{transcript}\n\n"
            "Return only the updated summary paragraph."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        new_summary = "".join(b.text for b in resp.content if b.type == "text").strip()
        new_upto = max(_id for _id, _r, _c in to_fold)
        save_state(chat_id, new_summary, new_upto)
        print(f"Summary updated for {chat_id}, upto id {new_upto}")
    except Exception as e:
        print("Summary update failed (non-fatal):", e)


def build_system_prompt(chat_id: str) -> str:
    """STYLE_PROMPT plus current time, the person's long-term facts and rolling summary."""
    parts = [STYLE_PROMPT]
    parts.append(f"\nCURRENT DATE & TIME (their local time): {now_local_str()}")
    facts = load_facts(chat_id)
    if facts:
        parts.append("\nLONG-TERM things you know about them:\n" + "\n".join(f"- {f}" for f in facts))
    summary, _ = load_state(chat_id)
    if summary:
        parts.append("\nSUMMARY of your earlier conversations:\n" + summary)
    return "\n".join(parts)


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


def count_user_messages(chat_id: str) -> int:
    if not DATABASE_URL:
        return 0
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM messages WHERE chat_id = %s AND role = 'user'",
            (chat_id,),
        )
        return cur.fetchone()[0]


def generate_reply(chat_id: str, user_content, text_for_storage: str) -> list:
    history = load_recent_history(chat_id, limit=RECENT_TURNS)
    messages = history + [{"role": "user", "content": user_content}]
    system = build_system_prompt(chat_id)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=system,
        messages=messages,
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip().strip('"')

    # Persist this exchange.
    save_message(chat_id, "user", text_for_storage)
    save_message(chat_id, "assistant", reply)

    # Only run fact extraction every 3rd user message, to cut latency/cost.
    # Counted from the DB so it's stable across restarts.
    if count_user_messages(chat_id) % 3 == 0:
        existing_facts = load_facts(chat_id)
        extract_and_store_facts(chat_id, text_for_storage, reply, existing_facts)

    # Summary folding only fires when enough backlog has built up anyway,
    # so it's already infrequent -- leave it running each turn.
    maybe_update_summary(chat_id)

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


def transcribe_voice(file_id: str):
    """Download a Telegram voice note and transcribe it with OpenAI Whisper.
    Returns the transcript text, or None if it fails / no key set."""
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY set -- can't transcribe voice notes.")
        return None
    try:
        # 1. get the file path from Telegram
        r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=20)
        file_path = r.json().get("result", {}).get("file_path")
        if not file_path:
            return None
        # 2. download the audio bytes (Telegram voice notes are .oga / opus)
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
        audio = requests.get(file_url, timeout=30).content

        # 3. send to Whisper
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("voice.oga", audio, "audio/ogg")},
            data={"model": "whisper-1"},
            timeout=60,
        )
        if resp.status_code != 200:
            print("Whisper error:", resp.status_code, resp.text)
            return None
        return resp.json().get("text", "").strip()
    except Exception as e:
        print("Transcription failed:", e)
        return None


@app.route("/", methods=["GET"])
def health():
    return "ok", 200


# ---------------------------------------------------------------------------
# Proactive check-ins
# ---------------------------------------------------------------------------
def get_last_checkin(chat_id: str):
    if not DATABASE_URL:
        return None
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT last_checkin FROM chat_state WHERE chat_id = %s", (chat_id,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def set_last_checkin(chat_id: str, when: datetime):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_state (chat_id, last_checkin) VALUES (%s, %s)
            ON CONFLICT (chat_id) DO UPDATE SET last_checkin = EXCLUDED.last_checkin
            """,
            (chat_id, when),
        )
        conn.commit()


def generate_checkin(chat_id: str) -> list:
    """Generate an unprompted check-in message, aware of context + history."""
    history = load_recent_history(chat_id, limit=RECENT_TURNS)
    system = build_system_prompt(chat_id)
    # A synthetic instruction telling the bot to reach out first.
    nudge = (
        "Reach out to them first, unprompted, like a friend randomly texting to "
        "check in. Keep it natural and short. If something's going on in their "
        "life that you know about, you can reference it. Don't say you're an AI "
        "or that this is automated. Just text them like a friend would."
    )
    messages = history + [{"role": "user", "content": nudge}]

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=system,
        messages=messages,
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip().strip('"')

    # Store the check-in as an assistant message so it's part of history.
    save_message(chat_id, "assistant", reply)

    bubbles = [b.strip() for b in reply.split("|||") if b.strip()]
    return bubbles or [reply]


@app.route("/checkin", methods=["GET", "POST"])
def checkin():
    # Protect the endpoint so randoms can't trigger messages to you.
    token = request.args.get("secret") or request.headers.get("X-Checkin-Secret")
    if not CHECKIN_SECRET or token != CHECKIN_SECRET:
        return "forbidden", 403

    if not ALLOWED_CHAT_ID:
        return "no ALLOWED_CHAT_ID set", 400

    # Roll the dice: only send sometimes, so timing feels random.
    if random.random() > CHECKIN_PROBABILITY:
        return "skipped (dice)", 200

    # Enforce a minimum gap since the last check-in.
    last = get_last_checkin(ALLOWED_CHAT_ID)
    now = datetime.now(timezone.utc)
    if last:
        hours_since = (now - last).total_seconds() / 3600
        if hours_since < CHECKIN_MIN_GAP_HOURS:
            return f"skipped (only {hours_since:.1f}h since last)", 200

    print("Sending proactive check-in")
    send_typing(ALLOWED_CHAT_ID)
    bubbles = generate_checkin(ALLOWED_CHAT_ID)
    for i, bubble in enumerate(bubbles):
        if i > 0:
            send_typing(ALLOWED_CHAT_ID)
            time.sleep(min(1.5, 0.4 + len(bubble) * 0.03))
        send_message(ALLOWED_CHAT_ID, bubble)

    set_last_checkin(ALLOWED_CHAT_ID, now)
    return "sent", 200


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
    voice = message.get("voice") or message.get("audio")

    image_b64 = media_type = None
    is_voice = False
    if photos:
        file_id = photos[-1]["file_id"]
        image_b64, media_type = download_telegram_image(file_id)
        text = caption
    elif voice:
        send_typing(chat_id)  # transcription takes a moment
        transcript = transcribe_voice(voice["file_id"])
        if transcript:
            text = transcript
            is_voice = True
        else:
            send_message(chat_id, "couldnt make out that voice note 😭 mind typing it")
            return "ok", 200

    if not text and not image_b64:
        return "ok", 200

    print(f"Incoming from {chat_id}: text={text!r} image={'yes' if image_b64 else 'no'} voice={is_voice}")
    send_typing(chat_id)

    user_content = build_user_content(text, image_b64, media_type)
    if image_b64:
        stored_text = text if text else "[sent a photo]"
    elif is_voice:
        stored_text = f"[voice note] {text}"  # keep a marker so history shows it was spoken
    else:
        stored_text = text
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
