# mini_ai_mentor.py

import threading
from pynput import keyboard
from collections import deque, defaultdict
from .utils import speak
import time

class MiniAIMentor:
    """
    Mini AI Mentor that runs in background,
    detects repeated commands/actions, and
    triggers suggestions via a custom speak() function.
    """
    def __init__(self, repetition_threshold=3, max_history=10):
        self.command_history = deque(maxlen=max_history)
        self.command_count = defaultdict(int)
        self.repetition_threshold = repetition_threshold
        self.typed_command = ""
        self.jarvis_popup_active = False  # True when popup is active
        self.listener = None

    # -------------------------
    # Popup state control
    # -------------------------
    def set_popup_state(self, state: bool):
        """
        Call this from main code to pause/resume detection
        when Jarvis popup opens/closes
        """
        self.jarvis_popup_active = state

    # -------------------------
    # Command processing
    # -------------------------
    def process_command(self, command):
        command = command.strip()
        if not command:
            return

        self.command_history.append(command)
        self.command_count[command] += 1

        if self.command_count[command] == self.repetition_threshold:
            self.suggest(command)

    def suggest(self, command):
        """
        Trigger a suggestion via custom speak() function
        """
        tip = self.generate_tip(command)
        # This will use your imported speak() function
        # Example: speak(f"Hey! I noticed you repeated '{command}' several times. {tip}")
        self.on_suggestion(command, tip)

    def generate_tip(self, command):
        """
        Dynamically generate tips based on command patterns
        """
        if command.startswith("git "):
            return "Consider using aliases or combined commands to save time."
        elif command.startswith("mkdir"):
            return "You can create multiple directories at once by listing them together."
        elif command.startswith("cd"):
            return "Use absolute paths or aliases to jump directly to frequently used folders."
        elif command.startswith("ls"):
            return "Add flags like '-lh' for human-readable sizes or '-la' to see hidden files."
        else:
            return "Try automating repetitive steps or creating a shortcut for this command."

    # -------------------------
    # Suggestion callback (override or assign)
    # -------------------------
    def on_suggestion(self, command, tip):
        """
        Override this method or assign a callback in main code to handle suggestions.
        Default: print to console
        """
        print(f"[Suggestion] {command}: {tip}")
        speak(f"[Suggestion] {command}: {tip}")

    # -------------------------
    # Keyboard Listener
    # -------------------------
    def _on_press(self, key):
        if self.jarvis_popup_active:
            return  # Pause detection when popup is active

        try:
            if key.char:
                self.typed_command += key.char
        except AttributeError:
            if key == keyboard.Key.enter:
                self.process_command(self.typed_command)
                self.typed_command = ""
            elif key == keyboard.Key.backspace:
                self.typed_command = self.typed_command[:-1]

    def start_listener(self):
        """
        Start keyboard listener in background
        """
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def run_background(self):
        """
        Start the mentor as a background thread
        """
        self.start_listener()
        while True:
            time.sleep(1)  # Keep thread alive
