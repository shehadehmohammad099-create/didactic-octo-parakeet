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
import threading
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

# ---- Human-like texting mechanics ----
REPLY_DELAY_MIN = float(os.environ.get("REPLY_DELAY_MIN", "2"))    # seconds
REPLY_DELAY_MAX = float(os.environ.get("REPLY_DELAY_MAX", "45"))   # seconds
TYPO_PROBABILITY = float(os.environ.get("TYPO_PROBABILITY", "0.12"))     # chance a reply includes a typo+fix
SILENCE_PROBABILITY = float(os.environ.get("SILENCE_PROBABILITY", "0.04"))  # chance it leaves you on delivered
BATCH_WAIT = float(os.environ.get("BATCH_WAIT", "8"))  # seconds to wait for more messages before replying

# Message batching: rapid-fire messages pool here and get answered as one.
_PENDING = {}  # chat_id -> {"buffer": [texts], "seq": int}
_PENDING_LOCK = threading.Lock()
BATCH_WINDOW = float(os.environ.get("BATCH_WINDOW", "10"))  # seconds to wait for follow-up messages before replying

# Buffer for message batching: when you send several messages in a burst, they
# accumulate here and get processed as ONE message once you stop typing.
_PENDING = {}          # chat_id -> {"texts": [...], "image": (b64, media_type) or None, "timer": Timer}
_PENDING_LOCK = threading.Lock()

# Dedup Telegram update retries (needed because we ack instantly and reply slowly)
_SEEN_UPDATES = set()
_SEEN_LOCK = threading.Lock()

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

# ---- Version info ----
BOT_VERSION = "1.0 'inner life'"  # bump this when you feel like it
GIT_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "")[:7]  # auto-set by Render per deploy
STARTED_AT = datetime.now(timezone.utc)

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
- YOU HAVE YOUR OWN LIFE: you have a mood, an energy level, things you're into
  (you'll be told your current state). Let it color your texting. Sometimes bring up
  your own stuff unprompted ("ok random but"). Hold consistent opinions and
  preferences over time -- you actually like and dislike things, and you don't just
  mirror them.
- EMOTIONAL CARRYOVER: if they told you something heavy recently, carry that weight.
  Don't reset to cheerful next message. Check in on it, be a bit gentler, let it
  linger the way it would for a real friend.
- You don't have to address EVERYTHING they said. Real friends sometimes react to
  just one part, or reply with a single word or emoji when that's the honest reaction.
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
        cur.execute("ALTER TABLE chat_state ADD COLUMN IF NOT EXISTS last_reflected_id INTEGER DEFAULT 0")

        # The bot's private reflections between conversations (its "inner life").
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inner_thoughts (
                id       SERIAL PRIMARY KEY,
                chat_id  TEXT NOT NULL,
                thought  TEXT NOT NULL,
                ts       TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_thoughts_chat ON inner_thoughts (chat_id, id)")

        # The bot's own evolving state (mood, energy, current interest).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_state (
                chat_id   TEXT PRIMARY KEY,
                mood      TEXT DEFAULT 'content',
                energy    TEXT DEFAULT 'medium',
                interest  TEXT DEFAULT '',
                updated   TIMESTAMPTZ DEFAULT now()
            )
            """
        )

        # Episodic memories: discrete meaningful moments/arcs with weight.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id           SERIAL PRIMARY KEY,
                chat_id      TEXT NOT NULL,
                title        TEXT NOT NULL,
                memory       TEXT NOT NULL,
                significance INTEGER DEFAULT 5,   -- 1-10
                ts           TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_chat ON episodes (chat_id, significance)")

        # Upcoming plans/events extracted from conversation, for proactive check-ins.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id          SERIAL PRIMARY KEY,
                chat_id     TEXT NOT NULL,
                description TEXT NOT NULL,
                due_date    DATE,
                mentioned   BOOLEAN DEFAULT FALSE,
                ts          TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plans_chat ON plans (chat_id, due_date)")

        # People in their life, so the bot can ask about them by name.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id       SERIAL PRIMARY KEY,
                chat_id  TEXT NOT NULL,
                name     TEXT NOT NULL,
                relation TEXT DEFAULT '',
                notes    TEXT DEFAULT '',
                updated  TIMESTAMPTZ DEFAULT now(),
                UNIQUE (chat_id, name)
            )
            """
        )
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


# ---- Inner life (private thoughts) ----
def load_recent_thoughts(chat_id: str, limit: int = 3) -> list:
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT thought, ts FROM inner_thoughts WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
            (chat_id, limit),
        )
        rows = cur.fetchall()
    out = []
    for thought, ts in reversed(rows):
        local = ts.astimezone(LOCAL_TZ) if ts else None
        stamp = local.strftime("%a") if local else ""
        out.append(f"({stamp}) {thought}" if stamp else thought)
    return out


def add_thought(chat_id: str, thought: str):
    if not DATABASE_URL or not thought:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO inner_thoughts (chat_id, thought) VALUES (%s, %s)", (chat_id, thought))
        conn.commit()


# ---- Bot's own state ----
def load_bot_state(chat_id: str) -> dict:
    if not DATABASE_URL:
        return {"mood": "content", "energy": "medium", "interest": ""}
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT mood, energy, interest FROM bot_state WHERE chat_id = %s", (chat_id,))
        row = cur.fetchone()
    if row:
        return {"mood": row[0], "energy": row[1], "interest": row[2]}
    return {"mood": "content", "energy": "medium", "interest": ""}


def save_bot_state(chat_id: str, mood: str, energy: str, interest: str):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bot_state (chat_id, mood, energy, interest, updated)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (chat_id) DO UPDATE
              SET mood = EXCLUDED.mood, energy = EXCLUDED.energy,
                  interest = EXCLUDED.interest, updated = now()
            """,
            (chat_id, mood, energy, interest),
        )
        conn.commit()


# ---- Episodic memory ----
def load_top_episodes(chat_id: str, limit: int = 5) -> list:
    """Most significant episodes, recent ones favored on ties."""
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, memory, ts FROM episodes
            WHERE chat_id = %s
            ORDER BY significance DESC, id DESC
            LIMIT %s
            """,
            (chat_id, limit),
        )
        rows = cur.fetchall()
    out = []
    for title, memory, ts in rows:
        local = ts.astimezone(LOCAL_TZ) if ts else None
        when = local.strftime("%d %b") if local else ""
        out.append(f"[{when}] {title}: {memory}" if when else f"{title}: {memory}")
    return out


def add_episodes(chat_id: str, episodes: list):
    if not DATABASE_URL or not episodes:
        return
    with db_conn() as conn, conn.cursor() as cur:
        for ep in episodes:
            cur.execute(
                "INSERT INTO episodes (chat_id, title, memory, significance) VALUES (%s, %s, %s, %s)",
                (chat_id, ep.get("title", "moment"), ep.get("memory", ""), int(ep.get("significance", 5))),
            )
        conn.commit()


# ---- Plans (proactive memory of upcoming events) ----
def add_plans(chat_id: str, plans: list):
    if not DATABASE_URL or not plans:
        return
    with db_conn() as conn, conn.cursor() as cur:
        for p in plans:
            desc = p.get("description", "").strip()
            date_str = p.get("date")  # expected YYYY-MM-DD or null
            if not desc:
                continue
            # avoid duplicate plans with same description still pending
            cur.execute(
                "SELECT 1 FROM plans WHERE chat_id = %s AND description = %s AND mentioned = FALSE",
                (chat_id, desc),
            )
            if cur.fetchone():
                continue
            cur.execute(
                "INSERT INTO plans (chat_id, description, due_date) VALUES (%s, %s, %s)",
                (chat_id, desc, date_str),
            )
        conn.commit()


def get_due_plans(chat_id: str) -> list:
    """Plans due today (or overdue) that haven't been proactively mentioned yet."""
    if not DATABASE_URL:
        return []
    today = datetime.now(LOCAL_TZ).date()
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, description, due_date FROM plans
            WHERE chat_id = %s AND mentioned = FALSE AND due_date IS NOT NULL AND due_date <= %s
            ORDER BY due_date
            """,
            (chat_id, today),
        )
        return cur.fetchall()


def mark_plan_mentioned(plan_id: int):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE plans SET mentioned = TRUE WHERE id = %s", (plan_id,))
        conn.commit()


def load_upcoming_plans(chat_id: str, limit: int = 5) -> list:
    if not DATABASE_URL:
        return []
    today = datetime.now(LOCAL_TZ).date()
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT description, due_date FROM plans
            WHERE chat_id = %s AND (due_date IS NULL OR due_date >= %s) AND mentioned = FALSE
            ORDER BY due_date NULLS LAST LIMIT %s
            """,
            (chat_id, today, limit),
        )
        rows = cur.fetchall()
    out = []
    for desc, due in rows:
        out.append(f"{desc} ({due.strftime('%a %d %b')})" if due else desc)
    return out


# ---- People in their life ----
def upsert_people(chat_id: str, people: list):
    if not DATABASE_URL or not people:
        return
    with db_conn() as conn, conn.cursor() as cur:
        for p in people:
            name = p.get("name", "").strip()
            if not name:
                continue
            cur.execute(
                """
                INSERT INTO people (chat_id, name, relation, notes, updated)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (chat_id, name) DO UPDATE
                  SET relation = CASE WHEN EXCLUDED.relation <> '' THEN EXCLUDED.relation ELSE people.relation END,
                      notes = CASE WHEN EXCLUDED.notes <> '' THEN EXCLUDED.notes ELSE people.notes END,
                      updated = now()
                """,
                (chat_id, name, p.get("relation", ""), p.get("notes", "")),
            )
        conn.commit()


def load_people(chat_id: str, limit: int = 10) -> list:
    if not DATABASE_URL:
        return []
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, relation, notes FROM people WHERE chat_id = %s ORDER BY updated DESC LIMIT %s",
            (chat_id, limit),
        )
        rows = cur.fetchall()
    out = []
    for name, relation, notes in rows:
        line = name + (f" ({relation})" if relation else "")
        if notes:
            line += f" -- {notes}"
        out.append(line)
    return out


# ---- Reflection engine ----
def get_reflection_marker(chat_id: str) -> int:
    if not DATABASE_URL:
        return 0
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT last_reflected_id FROM chat_state WHERE chat_id = %s", (chat_id,))
        row = cur.fetchone()
    return row[0] if row and row[0] else 0


def set_reflection_marker(chat_id: str, upto_id: int):
    if not DATABASE_URL:
        return
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_state (chat_id, last_reflected_id) VALUES (%s, %s)
            ON CONFLICT (chat_id) DO UPDATE SET last_reflected_id = EXCLUDED.last_reflected_id
            """,
            (chat_id, upto_id),
        )
        conn.commit()


def reflect(chat_id: str):
    """The bot's between-conversation inner life. Runs on cron pings when there's
    been new conversation since the last reflection. One Claude call produces:
    a private thought, an updated mood/state, and any episodes worth keeping."""
    if not DATABASE_URL:
        return
    try:
        marker = get_reflection_marker(chat_id)
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, role, content FROM messages WHERE chat_id = %s AND id > %s ORDER BY id",
                (chat_id, marker),
            )
            new_msgs = cur.fetchall()

        if len(new_msgs) < 4:
            return  # not enough new conversation to reflect on

        transcript = "\n".join(f"{role}: {content}" for _id, role, content in new_msgs)
        state = load_bot_state(chat_id)
        recent_thoughts = load_recent_thoughts(chat_id, limit=3)

        prompt = (
            "You are the inner voice of someone's close friend, reflecting privately "
            "after recent conversations with them. Current time: " + now_local_str() + ".\n\n"
            f"Your current state: mood={state['mood']}, energy={state['energy']}, "
            f"currently into: {state['interest'] or 'nothing in particular'}\n"
            f"Your recent private thoughts:\n" + ("\n".join(recent_thoughts) or "(none)") + "\n\n"
            f"The recent conversation:\n{transcript}\n\n"
            "Reflect as their friend would. Respond ONLY with JSON, no markdown fences:\n"
            "{\n"
            '  "thought": "a short private reflection in casual first person -- what stayed with you, '
            'what you\'re wondering about them, anything you want to follow up on",\n'
            '  "mood": "one or two words for your own current mood (let it drift naturally, '
            'influenced a little by the conversations but also just life)",\n'
            '  "energy": "low, medium, or high",\n'
            '  "interest": "a thing you\'re personally into at the moment (a show, an idea, a hobby '
            '-- invent/evolve this naturally, it\'s YOUR life)",\n'
            '  "episodes": [ {"title": "...", "memory": "1-2 sentence memory of a meaningful moment '
            'or arc from these conversations", "significance": 1-10} ],\n'
            '  "plans": [ {"description": "an upcoming event/plan they mentioned (exam, trip, match, '
            'appointment)", "date": "YYYY-MM-DD or null if unknown"} ],\n'
            '  "people": [ {"name": "person they mentioned", "relation": "friend/mum/teacher/etc if '
            'clear", "notes": "current situation with them, brief"} ]\n'
            "}\n"
            "episodes should usually be empty [] -- only add one for genuinely meaningful moments "
            "(big news, emotional conversations, decisions, memorable jokes), not routine chat. "
            "plans: only real, dated-ish future events (resolve relative dates like 'thursday' to an "
            "actual date using the current time given above). people: only actual named people in "
            "their life, not celebrities."
        )

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(b.text for b in resp.content if b.type == "text").strip()
        out = out.replace("```json", "").replace("```", "").strip()

        import json as _json
        data = _json.loads(out)

        if data.get("thought"):
            add_thought(chat_id, data["thought"])
        save_bot_state(
            chat_id,
            data.get("mood", state["mood"]),
            data.get("energy", state["energy"]),
            data.get("interest", state["interest"]),
        )
        add_episodes(chat_id, data.get("episodes", []))
        add_plans(chat_id, data.get("plans", []))
        upsert_people(chat_id, data.get("people", []))

        set_reflection_marker(chat_id, max(m[0] for m in new_msgs))
        print(f"Reflected for {chat_id}: mood={data.get('mood')}, "
              f"{len(data.get('episodes', []))} episode(s), thought saved.")
    except Exception as e:
        print("Reflection failed (non-fatal):", e)


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
    """STYLE_PROMPT plus current time, the bot's own state/inner life, and memory layers."""
    parts = [STYLE_PROMPT]
    parts.append(f"\nCURRENT DATE & TIME (their local time): {now_local_str()}")

    # The bot's own life
    state = load_bot_state(chat_id)
    parts.append(
        f"\nYOUR OWN current state (let it subtly color your texting, don't announce it): "
        f"mood: {state['mood']}, energy: {state['energy']}"
        + (f", currently into: {state['interest']}" if state['interest'] else "")
    )
    thoughts = load_recent_thoughts(chat_id, limit=3)
    if thoughts:
        parts.append(
            "\nYOUR recent private thoughts about them (things that stayed with you between chats "
            "-- you can naturally follow up on these, like a friend who's been thinking):\n"
            + "\n".join(f"- {t}" for t in thoughts)
        )

    # Memory of them
    facts = load_facts(chat_id)
    if facts:
        parts.append("\nLONG-TERM things you know about them:\n" + "\n".join(f"- {f}" for f in facts))
    people = load_people(chat_id)
    if people:
        parts.append(
            "\nPEOPLE in their life (ask about them BY NAME when it fits -- 'hows things with X'):\n"
            + "\n".join(f"- {p}" for p in people)
        )
    plans = load_upcoming_plans(chat_id)
    if plans:
        parts.append(
            "\nTHEIR UPCOMING PLANS (be aware of these -- wish luck before, ask how it went after):\n"
            + "\n".join(f"- {p}" for p in plans)
        )
    episodes = load_top_episodes(chat_id, limit=5)
    if episodes:
        parts.append(
            "\nSHARED MEMORIES (specific moments you both lived through -- reference them "
            "naturally when relevant, like 'remember when...'):\n"
            + "\n".join(f"- {e}" for e in episodes)
        )
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


def generate_reply(chat_id: str, user_content, text_for_storage: str, typo_turn: bool = False) -> list:
    history = load_recent_history(chat_id, limit=RECENT_TURNS)
    messages = history + [{"role": "user", "content": user_content}]
    system = build_system_prompt(chat_id)
    if typo_turn:
        system += (
            "\n\nTHIS REPLY ONLY: include one small realistic typo in one of your "
            "bubbles, then send a quick correction as its own bubble (like 'fecning' "
            "then 'fencing*' or 'wait no i mean...'). Keep it natural, not forced."
        )

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
    # A synthetic instruction telling the bot to reach out first. ~30% of the
    # time it leads with its own life instead of checking in on them.
    if random.random() < 0.3:
        nudge = (
            "Text them first, unprompted -- but this time lead with YOUR own stuff: "
            "something you've been thinking about, into lately, or that 'happened' in "
            "your day (use your current state/interest). Like a friend going 'ok random "
            "but...'. Keep it natural and short. Don't say you're an AI or automated."
        )
    else:
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


def run_checkin_work(force: bool):
    """All the check-in work, run in a background thread so the endpoint can
    respond to the cron pinger instantly (avoids timeouts / retries)."""
    try:
        # Reflect first: the bot's inner life runs on every ping, whether or not
        # a message gets sent. This is what makes follow-ups feel self-motivated.
        reflect(ALLOWED_CHAT_ID)

        # PLAN-DUE check-ins take priority and skip the dice: if something they
        # mentioned is happening today, the bot reaches out about it specifically.
        due = get_due_plans(ALLOWED_CHAT_ID)
        if due:
            plan_id, description, due_date = due[0]
            print(f"Plan due today: {description!r} -- sending targeted check-in")
            send_typing(ALLOWED_CHAT_ID)
            history = load_recent_history(ALLOWED_CHAT_ID, limit=RECENT_TURNS)
            system = build_system_prompt(ALLOWED_CHAT_ID)
            nudge = (
                f"You remember that TODAY they have this: \"{description}\". Text them "
                "first about it specifically -- wish them luck, ask if they're ready, or "
                "ask how it went if it's likely already over given the current time. "
                "Natural and short, like a friend who remembered. Don't say you're an AI."
            )
            resp2 = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                system=system,
                messages=history + [{"role": "user", "content": nudge}],
            )
            reply = "".join(b.text for b in resp2.content if b.type == "text").strip().strip('"')
            save_message(ALLOWED_CHAT_ID, "assistant", reply)
            for i, bubble in enumerate([b.strip() for b in reply.split("|||") if b.strip()]):
                if i > 0:
                    send_typing(ALLOWED_CHAT_ID)
                    time.sleep(min(1.5, 0.4 + len(bubble) * 0.03))
                send_message(ALLOWED_CHAT_ID, bubble)
            mark_plan_mentioned(plan_id)
            set_last_checkin(ALLOWED_CHAT_ID, datetime.now(timezone.utc))
            print("Check-in result: sent (plan due)")
            return

        # Roll the dice: only send sometimes, so timing feels random.
        if not force and random.random() > CHECKIN_PROBABILITY:
            print("Check-in skipped: dice roll")
            return

        # Enforce a minimum gap since the last check-in.
        last = get_last_checkin(ALLOWED_CHAT_ID)
        now = datetime.now(timezone.utc)
        if last and not force:
            hours_since = (now - last).total_seconds() / 3600
            if hours_since < CHECKIN_MIN_GAP_HOURS:
                print(f"Check-in skipped: only {hours_since:.1f}h since last")
                return

        print("Sending proactive check-in")
        send_typing(ALLOWED_CHAT_ID)
        bubbles = generate_checkin(ALLOWED_CHAT_ID)
        for i, bubble in enumerate(bubbles):
            if i > 0:
                send_typing(ALLOWED_CHAT_ID)
                time.sleep(min(1.5, 0.4 + len(bubble) * 0.03))
            send_message(ALLOWED_CHAT_ID, bubble)

        set_last_checkin(ALLOWED_CHAT_ID, now)
        print("Check-in result: sent")
    except Exception as e:
        print("Check-in work failed:", e)


@app.route("/checkin", methods=["GET", "POST"])
def checkin():
    # Protect the endpoint so randoms can't trigger messages to you.
    token = request.args.get("secret") or request.headers.get("X-Checkin-Secret")
    if not CHECKIN_SECRET or token != CHECKIN_SECRET:
        return "forbidden", 403

    if not ALLOWED_CHAT_ID:
        return "no ALLOWED_CHAT_ID set", 400

    force = request.args.get("force") == "1"

    # Respond to the pinger IMMEDIATELY with a tiny body; do everything in a
    # background thread. Outcomes are visible in Render logs.
    threading.Thread(target=run_checkin_work, args=(force,), daemon=True).start()
    return "ok", 200


def human_delay_for(text_len: int) -> float:
    """A believable 'saw it, then replied' delay. Short messages sometimes get
    fast reactions; sometimes life gets in the way."""
    base = random.uniform(REPLY_DELAY_MIN, REPLY_DELAY_MAX * 0.4)
    if random.random() < 0.25:  # occasionally distracted
        base = random.uniform(REPLY_DELAY_MAX * 0.5, REPLY_DELAY_MAX)
    return base


def process_batch(chat_id, seq: int, image_b64=None, media_type=None):
    """Debounced batch processor: waits BATCH_WAIT; if newer messages arrived,
    bails (their thread will handle the whole buffer). Otherwise replies to the
    pooled burst of messages as ONE thought."""
    try:
        time.sleep(BATCH_WAIT)
        with _PENDING_LOCK:
            pending = _PENDING.get(str(chat_id))
            if not pending or pending["seq"] != seq:
                return  # newer message arrived; that thread owns the batch now
            texts = pending["buffer"][:]
            _PENDING.pop(str(chat_id), None)

        combined = "\n".join(t for t in texts if t).strip()
        if not combined and not image_b64:
            return

        # Occasionally leave them on delivered -- but still remember.
        if random.random() < SILENCE_PROBABILITY:
            print("Staying silent this time (still storing message)")
            save_message(str(chat_id), "user", combined or "[sent a photo]")
            return

        # Human-like pause (batch wait already added some).
        delay = max(0, human_delay_for(len(combined)) - BATCH_WAIT)
        if delay > 0:
            print(f"Waiting {delay:.0f}s more before replying (human latency)")
            time.sleep(delay)
        send_typing(chat_id)
        time.sleep(random.uniform(1.0, 3.0))

        user_content = build_user_content(combined, image_b64, media_type)
        stored_text = combined if combined else "[sent a photo]"

        typo_turn = random.random() < TYPO_PROBABILITY
        bubbles = generate_reply(str(chat_id), user_content, stored_text, typo_turn=typo_turn)

        for i, bubble in enumerate(bubbles):
            if i > 0:
                send_typing(chat_id)
                time.sleep(min(2.5, 0.4 + len(bubble) * 0.04))
            send_message(chat_id, bubble)
    except Exception as e:
        print("Batch processing failed:", e)


def queue_message(chat_id, text: str, image_b64=None, media_type=None):
    """Add a message to the chat's pending buffer and (re)start the debounce."""
    key = str(chat_id)
    with _PENDING_LOCK:
        entry = _PENDING.setdefault(key, {"buffer": [], "seq": 0})
        if text:
            entry["buffer"].append(text)
        entry["seq"] += 1
        seq = entry["seq"]
    threading.Thread(
        target=process_batch,
        args=(chat_id, seq, image_b64, media_type),
        daemon=True,
    ).start()


@app.route("/webhook", methods=["POST"])
def webhook():
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        return "forbidden", 403

    update = request.get_json(silent=True) or {}

    # Dedup: because we ack fast and reply slow, Telegram may re-deliver the
    # same update if anything hiccups. Never process the same update twice.
    update_id = update.get("update_id")
    if update_id is not None:
        with _SEEN_LOCK:
            if update_id in _SEEN_UPDATES:
                return "ok (duplicate)", 200
            _SEEN_UPDATES.add(update_id)
            if len(_SEEN_UPDATES) > 500:  # keep the set bounded
                for _ in range(100):
                    _SEEN_UPDATES.pop()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return "ok", 200

    chat_id = message["chat"]["id"]

    if ALLOWED_CHAT_ID and str(chat_id) != ALLOWED_CHAT_ID:
        print(f"Ignoring message from chat {chat_id} (not allowed)")
        return "ok", 200

    text = message.get("text", "")

    # ---- Commands (handled directly and instantly, never sent to Claude) ----
    if text.strip().lower() in ("/version", "/botversion", "/botversionnow", "/v"):
        state = load_bot_state(str(chat_id))
        uptime_h = (datetime.now(timezone.utc) - STARTED_AT).total_seconds() / 3600
        lines = [
            f"🤖 version {BOT_VERSION}" + (f" ({GIT_COMMIT})" if GIT_COMMIT else ""),
            f"⏱ this instance up {uptime_h:.1f}h (sleeps when idle, that's normal)",
            f"🧠 memory: {'ON (postgres)' if DATABASE_URL else 'OFF'}",
            f"🎙 voice notes: {'ON' if OPENAI_API_KEY else 'OFF (no OPENAI_API_KEY)'}",
            f"⏰ timezone: {TIMEZONE} — thinks it's {now_local_str()}",
            f"💭 current mood: {state['mood']}, energy {state['energy']}"
            + (f", into: {state['interest']}" if state['interest'] else ""),
        ]
        send_message(chat_id, "\n".join(lines))
        return "ok", 200

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

    # Queue into the debounce buffer: rapid multi-message bursts get pooled and
    # answered as ONE thought once you stop typing. Ack Telegram immediately.
    queued_text = f"[voice note] {text}" if is_voice else text
    queue_message(chat_id, queued_text, image_b64, media_type)

    return "ok", 200


# Initialize the DB table when the app starts (works under gunicorn too).
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
