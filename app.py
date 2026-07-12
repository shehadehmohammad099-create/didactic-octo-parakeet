#!/usr/bin/env python3
"""
Two-way Telegram bot that replies in your texting style using Claude.
Runs on Render (free) using webhook mode.

Environment variables (set in Render dashboard):
    ANTHROPIC_API_KEY   - your Claude API key
    TELEGRAM_TOKEN      - the token @BotFather gives you
    WEBHOOK_SECRET      - any random string you make up (used to verify that
                          incoming requests really come from Telegram)
    ALLOWED_CHAT_ID     - (optional) your numeric Telegram chat id. If set, the
                          bot only replies to you. Leave blank at first, then
                          fill it in once you've messaged the bot (see README).
"""

import os

import anthropic
import requests
from flask import Flask, request

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
ALLOWED_CHAT_ID = os.environ.get("ALLOWED_CHAT_ID", "").strip()

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

STYLE_PROMPT = """You are replying as a friend in a casual chat. Match this exact texting style:

- Keep it SHORT: 3-6 words average. Rarely more than one short sentence.
- Mostly lowercase, especially at the start. Skip ending punctuation (no periods).
- Use "u" and "ur" instead of "you"/"your".
- Occasionally use abbreviations naturally: ig, abt, js, tmrw, idk, nvm.
- Rarely use "!" -- but when you do, double it up ("!!").
- Occasional ALL CAPS for a sudden spike of excitement/shock/exasperation.
- 😭 is the go-to emoji, used sparingly.
- No corporate/assistant tone, no "I'm an AI" disclaimers. Reply like a real
  person texting back a friend -- reactive, casual, sometimes one word.
- Output ONLY the reply text, nothing else. No quotes around it.
"""

# In-memory history keyed by chat id. Resets if Render sleeps -- fine for casual use.
HISTORY = {}


def generate_reply(chat_id: str, incoming_text: str) -> str:
    convo = HISTORY.get(chat_id, [])
    convo.append({"role": "user", "content": incoming_text})

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        system=STYLE_PROMPT,
        messages=convo[-10:],
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip().strip('"')

    convo.append({"role": "assistant", "content": reply})
    HISTORY[chat_id] = convo[-20:]
    return reply


def send_message(chat_id, text: str) -> None:
    r = requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    print("Send status:", r.status_code, r.text)


@app.route("/", methods=["GET"])
def health():
    return "ok", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    # Verify the request actually came from Telegram using the secret header.
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        return "forbidden", 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")
    if not message:
        return "ok", 200

    chat_id = message["chat"]["id"]
    text = message.get("text", "")

    if not text:
        return "ok", 200  # ignore stickers/photos/etc for now

    if ALLOWED_CHAT_ID and str(chat_id) != ALLOWED_CHAT_ID:
        print(f"Ignoring message from chat {chat_id} (not allowed)")
        return "ok", 200

    print(f"Incoming from {chat_id}: {text}")
    reply = generate_reply(str(chat_id), text)
    print(f"Replying: {reply}")
    send_message(chat_id, reply)

    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
