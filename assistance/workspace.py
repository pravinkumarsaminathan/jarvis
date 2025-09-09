import datetime
import json
import os
import time
import psutil
import dateparser
import re

from .utils import speak

TIME_OF_DAY_WINDOWS = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 24)
}

TRACKED_APPS = {
    "code", "gnome-terminal", "qterminal", "xfce4-terminal", "chrome",
    "firefox", "firefox-bin", "firefox-esr", "spotify",
    "nautilus", "bash"
}

class JarvisWorkspaceManager:
    def __init__(self, memory_file="jarvis_memory.json", check_interval=5):
        self.memory_file = memory_file
        self.check_interval = check_interval

    def log_action(self, action: str, timestamp: str = None, details: dict = None):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
        entry = {"action": action, "timestamp": timestamp}
        if details:
            entry["details"] = details
        memory = self.load_memory()
        memory.append(entry)
        self.save_memory(memory)
        print(f"[Jarvis] Logged: {action} at {entry['timestamp']}" + (f" with {details}" if details else ""))

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            return []
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def save_memory(self, memory):
        with open(self.memory_file, "w") as f:
            json.dump(memory, f, indent=2)

    def monitor_apps(self):
        prev_procs = set()
        while True:
            current_procs = set()
            for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
                try:
                    pname = proc.info['name']
                    if pname and pname.lower() in TRACKED_APPS:
                        if pname.lower() == "code":
                            cmdline = proc.info.get('cmdline', [])
                            folder = None
                            for arg in cmdline[1:]:
                                if os.path.isdir(arg):
                                    folder = arg
                                    break
                            current_procs.add(("code", folder))
                        elif pname.lower() in {"qterminal", "gnome-terminal", "xfce4-terminal"}:
                            try:
                                cwd = proc.cwd()
                            except Exception:
                                cwd = None
                            current_procs.add((pname.lower(), cwd))
                        elif pname.lower() == "firefox-esr":
                            current_procs.add(("firefox-esr", None))
                        else:
                            current_procs.add((pname.lower(), None))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            opened = current_procs - prev_procs
            for app, detail in opened:
                if app == "code":
                    self.log_action(f"Opened code", details={"folder": detail} if detail else {})
                elif app in {"qterminal", "gnome-terminal", "xfce4-terminal"}:
                    self.log_action(f"Opened {app}", details={"cwd": detail} if detail else {})
                elif app == "firefox-esr":
                    self.log_action(f"Opened firefox-esr")
                else:
                    self.log_action(f"Opened {app}")

            closed = prev_procs - current_procs
            for app, detail in closed:
                self.log_action(f"Closed {app}")

            prev_procs = current_procs
            time.sleep(self.check_interval)

    def parse_human_query(self, query):
        # Extract time-of-day and date from human string
        query = query.lower()
        tod_phrase = None
        for phrase in TIME_OF_DAY_WINDOWS:
            if phrase in query:
                tod_phrase = phrase
                break

        # Extract date phrase (e.g., "yesterday", "last friday", "monday")
        date_match = re.search(r"(yesterday|today|last\s+\w+|friday|monday|tuesday|wednesday|thursday|saturday|sunday)", query)
        date_phrase = date_match.group(0) if date_match else ""

        # Compose NLP query
        nlp_query = f"{date_phrase} {tod_phrase}" if tod_phrase else date_phrase
        if not nlp_query.strip():
            nlp_query = "last night"  # fallback

        return nlp_query.strip()

    def get_workspace_for_time(self, query):
        # Extract time-of-day if present
        time_window = (0, 23)
        tod_phrase = None
        for phrase, hours in TIME_OF_DAY_WINDOWS.items():
            if phrase in query:
                time_window = hours
                tod_phrase = phrase
                query = query.replace(phrase, "").strip()
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
        memory = self.load_memory()
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

    def restore_workspace(self, human_query):
        nlp_query = self.parse_human_query(human_query)
        apps = self.get_workspace_for_time(nlp_query)
        speak(f"Restoring workspace for {nlp_query}")
        print(f"Setting up workspace for '{human_query}' ({nlp_query}): {', '.join(apps.keys())}")
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