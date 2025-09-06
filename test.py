import sqlite3
import os
import psutil
import time
import subprocess
from datetime import datetime
from collections import Counter

# Optional voice (comment out if not needed)
try:
    import pyttsx3
    TTS_ENABLED = True
    engine = pyttsx3.init()
except ImportError:
    TTS_ENABLED = False

DB_PATH = "state.db"
HIST_FILE = os.path.expanduser("~/.bash_history")
last_seen_command = None
last_seen_apps = set()


# ------------------------
# INIT DATABASE
# ------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hour INTEGER,
            command TEXT,
            context TEXT
        )
    """)
    try:
        c.execute("ALTER TABLE events ADD COLUMN hour INTEGER;")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


# ------------------------
# NOTIFY METHODS
# ------------------------
def notify(msg: str):
    """Desktop popup notification"""
    try:
        subprocess.run(["notify-send", "Jarvis Whisper", msg])
    except Exception as e:
        print(f"[notify error] {e}")


def speak(msg: str):
    """Voice output"""
    if TTS_ENABLED:
        engine.say(msg)
        engine.runAndWait()


# ------------------------
# DB + LEARNING
# ------------------------
def log_command(command):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hour = datetime.now().hour
    context = f"battery={psutil.sensors_battery().percent if psutil.sensors_battery() else -1}"
    c.execute("INSERT INTO events (command, context, hour) VALUES (?, ?, ?)", (command, context, hour))
    conn.commit()
    conn.close()


def predict_next(last_command, hour=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if hour is None:
        hour = datetime.now().hour

    c.execute("""
        SELECT e2.command
        FROM events e1
        JOIN events e2 ON e2.id = e1.id + 1
        WHERE e1.command = ?
        AND ABS(e1.hour - ?) <= 2
    """, (last_command, hour))

    results = [row[0] for row in c.fetchall()]
    conn.close()

    if not results:
        return None, 0.0

    counter = Counter(results)
    prediction, count = counter.most_common(1)[0]
    confidence = count / len(results)
    return prediction, confidence


# ------------------------
# SHELL HISTORY WATCHER
# ------------------------
def get_last_shell_command():
    """Read last line from bash history."""
    try:
        with open(HIST_FILE, "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else None
    except FileNotFoundError:
        return None


# ------------------------
# PROCESS WATCHER
# ------------------------
def get_running_apps():
    """Return a set of running GUI apps."""
    apps = set()
    for proc in psutil.process_iter(attrs=["name"]):
        try:
            name = proc.info["name"].lower()
            # focus only on common apps
            if name in ["firefox", "code", "vlc", "chromium", "gedit", "discord"]:
                apps.add(f"APP:{name}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return apps


# ------------------------
# BACKGROUND LOOP
# ------------------------
def background_loop():
    global last_seen_command, last_seen_apps
    init_db()
    print("🤖 Jarvis Whisper is running in background... (Ctrl+C to stop)")

    while True:
        # 1. Shell command tracking
        cmd = get_last_shell_command()
        if cmd and cmd != last_seen_command:
            last_seen_command = cmd
            log_command(cmd)
            prediction, conf = predict_next(cmd)
            if prediction:
                msg = f"After '{cmd}', you usually run '{prediction}' ({conf:.0%})"
                print(f"\n🤖 Jarvis Whisper: {msg}\n")
                notify(msg)
                speak(msg)
            else:
                msg = f"Learning new habit: '{cmd}'"
                print(f"\n🤖 Jarvis Whisper: {msg}\n")
                notify(msg)

        # 2. GUI app tracking (open + close)
        current_apps = get_running_apps()

        # Detect new launches
        new_apps = current_apps - last_seen_apps
        for app in new_apps:
            if app != last_seen_command:  # prevent loops
                last_seen_command = app
                log_command(app)
                prediction, conf = predict_next(app)
                if prediction:
                    msg = f"After '{app}', you usually run '{prediction}' ({conf:.0%})"
                    print(f"\n🤖 Jarvis Whisper: {msg}\n")
                    notify(msg)
                    speak(msg)
                else:
                    msg = f"Learning new habit: '{app}'"
                    print(f"\n🤖 Jarvis Whisper: {msg}\n")
                    notify(msg)

        # Detect closed apps
        closed_apps = last_seen_apps - current_apps
        for app in closed_apps:
            closed_event = f"{app}_closed"
            if closed_event != last_seen_command:
                last_seen_command = closed_event
                log_command(closed_event)
                prediction, conf = predict_next(closed_event)
                if prediction:
                    msg = f"After '{closed_event}', you usually run '{prediction}' ({conf:.0%})"
                    print(f"\n🤖 Jarvis Whisper: {msg}\n")
                    notify(msg)
                    speak(msg)
                else:
                    msg = f"Learning new habit: '{closed_event}'"
                    print(f"\n🤖 Jarvis Whisper: {msg}\n")
                    notify(msg)

        last_seen_apps = current_apps
        time.sleep(3)  # lightweight


if __name__ == "__main__":
    background_loop()
