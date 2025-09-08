import datetime
import json
import os
import time
import psutil

import dateparser
import datetime

MEMORY_FILE = "jarvis_memory.json"
CHECK_INTERVAL = 5  # seconds

# Map time-of-day phrases to hour ranges
TIME_OF_DAY_WINDOWS = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 24)
}

def log_action(action: str, timestamp: str = None, details: dict = None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
    entry = {"action": action, "timestamp": timestamp}
    if details:
        entry["details"] = details
    memory = load_memory()
    memory.append(entry)
    save_memory(memory)
    print(f"[Jarvis] Logged: {action} at {entry['timestamp']}" + (f" with {details}" if details else ""))

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# Only track these apps (add more as needed)
TRACKED_APPS = {
    "code", "gnome-terminal", "qterminal", "xfce4-terminal", "chrome",
    "firefox", "firefox-bin", "firefox-esr", "spotify",
    "nautilus", "bash"
}

def monitor_apps():
    prev_procs = set()
    while True:
        current_procs = set()
        for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
            try:
                pname = proc.info['name']
                if pname and pname.lower() in TRACKED_APPS:
                    # VS Code: get folder from cmdline
                    if pname.lower() == "code":
                        cmdline = proc.info.get('cmdline', [])
                        folder = None
                        for arg in cmdline[1:]:
                            if os.path.isdir(arg):
                                folder = arg
                                break
                        current_procs.add(("code", folder))
                    # Terminal: get cwd (not always possible, but try)
                    elif pname.lower() in {"qterminal", "gnome-terminal", "xfce4-terminal"}:
                        try:
                            cwd = proc.cwd()
                        except Exception:
                            cwd = None
                        current_procs.add((pname.lower(), cwd))
                    # Firefox: log process, but tabs need browser integration
                    elif pname.lower() == "firefox-esr":
                        current_procs.add(("firefox-esr", None))
                    else:
                        current_procs.add((pname.lower(), None))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Detect opened apps
        opened = current_procs - prev_procs
        for app, detail in opened:
            if app == "code":
                log_action(f"Opened code", details={"folder": detail} if detail else {})
            elif app in {"qterminal", "gnome-terminal", "xfce4-terminal"}:
                log_action(f"Opened {app}", details={"cwd": detail} if detail else {})
            elif app == "firefox-esr":
                log_action(f"Opened firefox-esr")
            else:
                log_action(f"Opened {app}")

        # Detect closed apps
        closed = prev_procs - current_procs
        for app, detail in closed:
            log_action(f"Closed {app}")

        prev_procs = current_procs
        time.sleep(CHECK_INTERVAL)

def get_last_night_workspace():
    memory = load_memory()
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    actions = [entry for entry in memory if entry["timestamp"].startswith(yesterday)]
    open_apps = {}
    for entry in actions:
        if entry["action"].startswith("Opened "):
            app = entry["action"].replace("Opened ", "")
            open_apps[app] = entry.get("details", {})
        elif entry["action"].startswith("Closed "):
            app = entry["action"].replace("Closed ", "")
            open_apps.pop(app, None)
    return open_apps

def setup_last_night_workspace():
    apps = get_last_night_workspace()
    print(f"Setting up last night's workspace: {', '.join(apps.keys())}")
    for app, details in apps.items():
        if app == "code":
            folder = details.get("folder", "/home/pravin/fesitval")
            os.system(f"code \"{folder}\" &")
        elif app == "qterminal":
            cwd = details.get("cwd", "/home/pravin")
            os.system(f"qterminal --workdir \"{cwd}\" &")
        elif app in {"gnome-terminal", "xfce4-terminal"}:
            cwd = details.get("cwd", "/home/pravin")
            os.system(f"{app} --working-directory=\"{cwd}\" &")
        elif app == "firefox-esr":
            # Advanced: restore tabs using session info or URLs
            urls = details.get("urls", [])
            if urls:
                for url in urls:
                    os.system(f"firefox-esr \"{url}\" &")
            else:
                os.system("firefox-esr &")
        elif app == "thunar":
            os.system("thunar &")
        # Add more elifs for other apps as needed

def get_workspace_for_time(query):
    # Extract time-of-day if present
    time_window = (0, 23)
    tod_phrase = None
    for phrase, hours in TIME_OF_DAY_WINDOWS.items():
        if phrase in query.lower():
            time_window = hours
            tod_phrase = phrase
            query = query.lower().replace(phrase, "").strip()
            break

    # If time-of-day was present and no hour in query, add a representative hour
    if tod_phrase and not any(char.isdigit() for char in query):
        mid_hour = (time_window[0] + time_window[1]) // 2
        query = f"{query} {mid_hour}:00"

    dt = dateparser.parse(query, settings={'RELATIVE_BASE': datetime.datetime.now()})
    print(f"Parsed datetime for '{query}': {dt}")
    if not dt:
        print("Could not understand the time in your query.")
        return {}

    date_str = dt.strftime("%Y-%m-%d")
    start_hour, end_hour = time_window
    memory = load_memory()
    open_apps = {}
    for entry in memory:
        ts = entry["timestamp"]
        if ts.startswith(date_str):
            try:
                entry_dt = datetime.datetime.strptime(ts, "%Y-%m-%d %I:%M %p")
                entry_hour = entry_dt.hour
                if start_hour <= entry_hour < end_hour:
                    if entry["action"].startswith("Opened "):
                        app = entry["action"].replace("Opened ", "")
                        open_apps[app] = entry.get("details", {})
                    elif entry["action"].startswith("Closed "):
                        app = entry["action"].replace("Closed ", "")
                        open_apps.pop(app, None)
            except Exception:
                continue
    return open_apps

def setup_workspace_for_time(query):
    apps = get_workspace_for_time(query)
    print(f"Setting up workspace for '{query}': {', '.join(apps.keys())}")
    for app, details in apps.items():
        if app == "code":
            folder = details.get("folder", "/home/pravin/fesitval")
            os.system(f"code \"{folder}\" &")
        elif app == "qterminal":
            cwd = details.get("cwd", "/home/pravin")
            os.system(f"qterminal --workdir \"{cwd}\" &")
        elif app in {"gnome-terminal", "xfce4-terminal"}:
            cwd = details.get("cwd", "/home/pravin")
            os.system(f"{app} --working-directory=\"{cwd}\" &")
        elif app == "firefox-esr":
            urls = details.get("urls", [])
            if urls:
                for url in urls:
                    os.system(f"firefox-esr \"{url}\" &")
            else:
                os.system("firefox-esr &")
        elif app == "thunar":
            os.system("thunar &")
        # Add more elifs for other apps as needed

# Example usage:
if __name__ == "__main__":
    print("Jarvis workflow monitor started...")
    # Uncomment below to test workspace setup
    setup_workspace_for_time("yesterday night")
    monitor_apps()
