import pyperclip, time, re, json, datetime, asyncio
from threading import Thread

BASE_DIR = __file__

# -----------------------------
# Storage
# -----------------------------
clipboard_history = []

# -----------------------------
# Categorizer
# -----------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", re.I)
ADDRESS_RE = re.compile(r"\d{1,5}\s+\w+\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Blvd|Block|Drive|Dr)\b", re.I)
FORMULA_RE = re.compile(r"^=.+")

CODE_LIKE_RE = re.compile(r"\b(pip |apt |sudo |git |python|SELECT |INSERT )")


def categorize_text(text):
    t = text.strip()
    if t.startswith("http://") or t.startswith("https://"):
        return "link", ["link"]
    elif EMAIL_RE.search(t):
        return "email", ["email"]
    elif PHONE_RE.search(t):
        return "phone", ["phone"]
    elif SQL_RE.search(t):
        return "sql", ["sql"]
    elif FORMULA_RE.match(t):
        return "formula", ["formula"]
    elif ADDRESS_RE.search(t):
        return "address", ["address"]
    elif CODE_LIKE_RE.search(t):
        return "command", ["command"]
    else:
        return "text", ["text"]


def add_clip(text):
    kind, tags = categorize_text(text)
    entry = {
        "content": text,
        "kind": kind,
        "tags": tags,
        "created_at": datetime.datetime.now().isoformat()
    }
    clipboard_history.append(entry)
    print(f"[+] Captured {kind}: {text[:60]}")

# -----------------------------
# Clipboard Monitor
# -----------------------------
last_clip = ""

def clipboard_loop():
    global last_clip
    print("ClipboardMonitor started.")
    try:
        while True:
            text = pyperclip.paste()
            if text and text != last_clip:
                last_clip = text
                add_clip(text)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped monitoring clipboard.")

# -----------------------------
# NLP Command Parser
# -----------------------------
def parse_command(text):
    t = text.lower()
    now = datetime.datetime.now()

    if "last" in t and "link" in t:
        return {"intent": "last_links"}
    elif "last copied one" in t:
        return {"intent": "last_one"}
    elif "list all" in t:
        return {"intent": "list_all"}
    elif "address" in t:
        return {"intent": "list_kind", "kind": "address"}
    elif "link" in t:
        return {"intent": "list_kind", "kind": "link"}
    elif "text" in t:
        return {"intent": "list_kind", "kind": "text"}
    elif "command" in t:
        return {"intent": "list_kind", "kind": "command"}
    elif "sql" in t:
        return {"intent": "list_kind", "kind": "sql"}
    elif "formula" in t:
        return {"intent": "list_kind", "kind": "formula"}
    elif "yesterday" in t:
        start = (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + datetime.timedelta(days=1)
        return {"intent": "time_filter", "start": start, "end": end}
    else:
        return {"intent": "last_n", "n": 3}

# -----------------------------
# Command Executor
# -----------------------------
async def execute_command(intent_data):
    intent = intent_data.get("intent")

    if intent == "last_links":
        links = [m for m in clipboard_history if m['kind'] == 'link'][-3:]
        if not links:
            return "No links found."
        return "\n".join([f"{i+1}. {l['content']}" for i, l in enumerate(links)])

    elif intent == "last_one":
        if clipboard_history:
            return clipboard_history[-1]['content']
        return "Clipboard empty."

    elif intent == "list_all":
        return "\n".join([f"[{m['kind']}] {m['content'][:50]}" for m in clipboard_history])

    elif intent == "list_kind":
        kind = intent_data.get("kind")
        items = [m for m in clipboard_history if m['kind']==kind]
        if not items:
            return f"No {kind}s found."
        return "\n".join([f"[{m['kind']}] {m['content']}" for m in items])

    elif intent == "time_filter":
        start = intent_data.get("start")
        end = intent_data.get("end")
        items = [m for m in clipboard_history if start <= datetime.datetime.fromisoformat(m['created_at']) < end]
        if not items:
            return "No items found in that time window."
        return "\n".join([f"[{m['kind']}] {m['content']}" for m in items])

    elif intent == "last_n":
        n = intent_data.get("n", 3)
        items = clipboard_history[-n:]
        if not items:
            return "Clipboard empty."
        return "\n".join([f"[{m['kind']}] {m['content']}" for m in items])

    return "Unknown command."

# -----------------------------
# Main CLI
# -----------------------------
def main():
    t = Thread(target=clipboard_loop, daemon=True)
    t.start()
    print("Jarvis Clipboard CLI — type natural commands or 'exit'")

    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            if q in ("exit", "quit"):
                break
            intent_data = parse_command(q)
            res = asyncio.get_event_loop().run_until_complete(execute_command(intent_data))
            print(res)
        except KeyboardInterrupt:
            break

    print("Exiting...")

if __name__ == "__main__":
    main()
