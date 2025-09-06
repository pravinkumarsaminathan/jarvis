from pynput import keyboard
from collections import deque, defaultdict
import pyttsx3

# -------------------------
# TTS Engine
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(line):
    print(f"Speaking: {line}")
    #engine.say(line)
    #engine.runAndWait()

# -------------------------
# Auto Mentor with Dynamic Tips
# -------------------------
class AutoMentorDynamic:
    def __init__(self, repetition_threshold=3, max_history=50):
        self.command_history = deque(maxlen=max_history)
        self.command_count = defaultdict(int)
        self.repetition_threshold = repetition_threshold

    def process_command(self, command):
        command = command.strip()
        if not command:
            return
        self.command_history.append(command)
        self.command_count[command] += 1
        if self.command_count[command] == self.repetition_threshold:
            self.suggest(command)

    def suggest(self, command):
        tip = self.generate_tip(command)
        speak(f"Hey! I noticed you repeated '{command}' several times. {tip}")

    def generate_tip(self, command):
        # Dynamic suggestion based on command patterns
        if command.startswith("git "):
            return "Consider using aliases or combined commands to save time."
        elif command.startswith("mkdir"):
            return "You can create multiple directories at once by listing them together."
        elif command.startswith("cd"):
            return "Use absolute paths or aliases to jump directly to frequently used folders."
        elif command.startswith("ls"):
            return "Add flags like '-lh' for human-readable sizes or '-la' to see hidden files."
        elif command.startswith("python") or command.endswith(".py"):
            return "You might want to run scripts with a single command or use a virtual environment."
        else:
            return "Try automating repetitive steps or creating a shortcut for this command."

# -------------------------
# Keyboard Listener
# -------------------------
mentor = AutoMentorDynamic()
typed_command = ""

def on_press(key):
    global typed_command
    try:
        if key.char:
            typed_command += key.char
    except AttributeError:
        if key == keyboard.Key.enter:
            mentor.process_command(typed_command)
            typed_command = ""
        elif key == keyboard.Key.backspace:
            typed_command = typed_command[:-1]

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Jarvis Mini AI Mentor running with dynamic tips... Press ESC to quit.")
listener.join()
