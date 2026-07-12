#!/usr/bin/env python3
"""
Two-way WhatsApp bot using the official WhatsApp Business Cloud API (Meta) +
Claude. Receives your messages via a webhook and replies in your texting style.

Environment variables (set in Render dashboard):
    ANTHROPIC_API_KEY   - your Claude API key
    WHATSAPP_TOKEN      - Meta access token (temporary 24h one to start, or a
                          permanent System User token for long-term use)
    PHONE_NUMBER_ID     - the "Phone number ID" from your Meta WhatsApp setup
    VERIFY_TOKEN        - any secret string you make up; must match what you
                          enter in the Meta webhook config
    ALLOWED_SENDER      - (optional) your WhatsApp number in international format
                          with no +, e.g. 447473073079. If set, the bot ONLY
                          replies to you and ignores anyone else.
"""

import os

import anthropic
import requests
from flask import Flask, request

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]
VERIFY_TOKEN = os.environ["VERIFY_TOKEN"]
ALLOWED_SENDER = os.environ.get("ALLOWED_SENDER", "").strip()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

STYLE_PROMPT = """You are replying as a friend in a casual WhatsApp conversation. Match this exact texting style:

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

# Very simple in-memory history keyed by sender number. Resets whenever Render
# spins the service down -- fine for casual use. For durable history you'd add
# a database, but that's overkill here.
HISTORY = {}


def generate_reply(sender: str, incoming_text: str) -> str:
    convo = HISTORY.get(sender, [])
    convo.append({"role": "user", "content": incoming_text})

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        system=STYLE_PROMPT,
        messages=convo[-10:],  # last 10 turns for context
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip().strip('"')

    convo.append({"role": "assistant", "content": reply})
    HISTORY[sender] = convo[-20:]
    return reply


def send_whatsapp(to: str, message: str) -> None:
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    print("Send status:", r.status_code, r.text)


@app.route("/", methods=["GET"])
def health():
    return "ok", 200


@app.route("/webhook", methods=["GET"])
def verify():
    # Meta calls this once to verify your webhook when you set it up.
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "verification failed", 403


@app.route("/webhook", methods=["POST"])
def incoming():
    data = request.get_json(silent=True) or {}
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        messages = entry.get("messages")
        if not messages:
            # could be a status update (delivered/read) -- ignore
            return "ok", 200

        msg = messages[0]
        sender = msg["from"]  # sender's number, no +
        text = msg.get("text", {}).get("body", "")

        if not text:
            return "ok", 200  # ignore non-text (images, etc.) for now

        if ALLOWED_SENDER and sender != ALLOWED_SENDER:
            print(f"Ignoring message from {sender} (not allowed sender)")
            return "ok", 200

        print(f"Incoming from {sender}: {text}")
        reply = generate_reply(sender, text)
        print(f"Replying: {reply}")
        send_whatsapp(sender, reply)

    except (KeyError, IndexError) as e:
        print("Couldn't parse webhook payload:", e, data)

    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
