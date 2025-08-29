import datetime
import os
import sys
import time
import webbrowser
import pyautogui
import pyttsx3
import speech_recognition as sr
import json
import random
import numpy as np
import psutil
import subprocess
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer
import re
import pywhatkit

# -------------------------------
# Tokenizer for vectorizer
# -------------------------------
def simple_tokenizer(txt):
    return txt.split()

add_safe_globals([CountVectorizer])  # safe unpickling

# -------------------------------
# Load trained chatbot model
# -------------------------------
data = torch.load("chat_model.pth", weights_only=False)
model_state = data["model_state"]
vectorizer = data["vectorizer"]
y_labels = data["y_labels"]
input_size = data["input_dim"]
hidden_size = data["hidden_dim"]
output_size = data["output_dim"]

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# -------------------------------
# Load intents
# -------------------------------
with open("data/intents.json") as file:
    intents = json.load(file)

# -------------------------------
# Load WhatsApp model
# -------------------------------
w_data = torch.load("greeting.pth", weights_only=False)
w_model_state = w_data["model_state"]
w_vectorizer = w_data["vectorizer"]
w_y_labels = w_data["y_labels"]
w_input_size = w_data["input_dim"]
w_hidden_size = w_data["hidden_dim"]
w_output_size = w_data["output_dim"]

class WhatsAppModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WhatsAppModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

w_model = WhatsAppModel(w_input_size, w_hidden_size, w_output_size)
w_model.load_state_dict(w_model_state)
w_model.eval()

with open("data/greeting.json") as file:
    greeting_intents = json.load(file)

# -------------------------------
# WhatsApp helper functions
# -------------------------------
def parse_send_message_command(cmd):
    cmd = cmd.lower()
    via_match = re.search(r'via (\w+)', cmd)
    via = via_match.group(1) if via_match else None
    name_match = re.search(r'to ([a-z]+)', cmd)
    name = name_match.group(1) if name_match else None
    msg_match = re.search(r'send (.+?) message to', cmd)
    if not msg_match:
        msg_match = re.search(r'send (.+?) to', cmd)
    message = msg_match.group(1).strip() if msg_match else None
    return {'name': name, 'message': message, 'via': via}

def load_contacts(filepath="data/contacts.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def send_whatsapp_message(phone_number, message):
    pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=10, tab_close=True)
    time.sleep(4)
    pyautogui.press("enter")
    pyautogui.hotkey("ctrl", "w")
    print("Bot: Message sent and tab closed.")

# -------------------------------
# Text-to-Speech
# -------------------------------
engine = pyttsx3.init("espeak")
engine.setProperty("voice", "english")
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text, speed=1.0, clarity=True):
    os.system(f'pico2wave -w temp.wav "{text}" > /dev/null 2>&1')
    cmd = "sox temp.wav temp_out.wav"
    if speed != 1.0:
        cmd += f" tempo {speed}"
    if clarity:
        cmd += " treble +5 highpass 200"
    cmd += " && aplay temp_out.wav > /dev/null 2>&1"
    os.system(cmd)

# -------------------------------
# Speech Recognition
# -------------------------------
def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = r.listen(source, phrase_time_limit=10)
    try:
        print("\rRecognition...")
        query = r.recognize_google(audio, language="en-in")
        print(f"\rUser said: {query}")
    except Exception:
        print("Say that again please...")
        return "None"
    return query.lower()

# -------------------------------
# Utility functions (day, wish, etc.)
# -------------------------------
def cal_day():
    day = datetime.datetime.today().weekday() + 1
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][day-1]

def wishMe():
    hour = int(datetime.datetime.now().hour)
    t = time.strftime("%I:%M:%p")
    day = cal_day()
    if hour < 12:
        speak(f"Good morning Pravin, it is {day} and the time is {t}")
    elif hour < 17:
        speak(f"Good afternoon Pravin, it is {day} and the time is {t}")
    else:
        speak(f'Good evening Pravin, it is {day} and the time is {t}')

# -------------------------------
# Functions (social, apps, etc.)
# -------------------------------
def social_media(cmd):
    sites = {
        "facebook": "https://www.facebook.com/",
        "whatsapp": "https://web.whatsapp.com/",
        "discord": "https://discord.com/",
        "instagram": "https://www.instagram.com/"
    }
    for key, url in sites.items():
        if key in cmd:
            speak(f"Opening your {key}")
            webbrowser.open(url)
            return
    speak("No result found")

def schedule():
    today = cal_day().lower()
    week = {
        "monday": "Boss, Monday starts strong! 9–9:50 Algorithms, then 10–11:50 System Design. Lunch at 12, and 2 PM Programming Lab.",
        "tuesday": "Boss, Tuesday’s packed! 9 Web Dev, 11 DB Systems. Lunch at 1, then Open Source Lab at 2.",
        "wednesday": "Boss, Wednesday vibes! 9 ML, 11 OS, 12 Ethics. Break at 1, and 2 PM Software Engg Workshop.",
        "thursday": "Boss, Thursday flow! 9 Networks, 11 Cloud Computing. Lunch at 1, then Cybersecurity Lab at 2.",
        "friday": "Boss, it’s Friday grind! 9 AI, 10 Adv Programming, 11 UI/UX. Lunch at 1, and Capstone work starts at 2.",
        "saturday": "Boss, Saturday schedule: morning Capstone meetings, noon Innovation session, and coding practice.",
        "sunday": "Boss, Sunday is chill mode! Rest or catch up on work."
    }
    speak(week.get(today, "No schedule found"), speed=0.8)

def openApp(cmd):
    if "calculator" in cmd:
        subprocess.Popen(["gnome-calculator"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(f"Opening calculator")
    elif "notepad" in cmd:
        subprocess.Popen(["gedit"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(f"Opening notepad")
    elif "paint" in cmd:
        subprocess.Popen(["gimp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(f"Opening paint")
    elif "google" in cmd:
        browsing()
    else:
        sites = {
        "facebook": "https://www.facebook.com/",
        "whatsapp": "https://web.whatsapp.com/",
        "discord": "https://discord.com/",
        "instagram": "https://www.instagram.com/"
        }
        for key, url in sites.items():
            if key in cmd:
                speak(f"Opening your {key}")
                webbrowser.open(url)
                return
        speak("No result found")

def closeApp(cmd):
    if "calculator" in cmd:
        speak(f"Closing calculator")
        os.system("pkill -f gnome-calculator > /dev/null 2>&1")
    elif "notepad" in cmd:
        speak(f"Closing notepad")
        os.system("pkill -f gedit > /dev/null 2>&1")
    elif "paint" in cmd:
        speak(f"Closing paint")
        os.system("pkill -f gimp > /dev/null 2>&1")
    else:
        speak("sorry i don't have access to do that.")

def browsing():
    speak("Boss, what should I search on Google?")
    s = command()
    if s != "None":
        webbrowser.open(f"https://www.google.com/search?q={s}")

def condition():
    usage = str(psutil.cpu_percent())
    speak(f"CPU is at {usage} percent")
    battery = psutil.sensors_battery()
    if battery:
        speak(f"Our system has {int(battery.percent)} percent battery")
    else:
        speak("I can't detect battery on this system")

def volume_control(action):
    if action == "up":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ +5% > /dev/null 2>&1")
        speak("Volume increased")
    elif action == "down":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ -5% > /dev/null 2>&1")
        speak("Volume decreased")

# -------------------------------
# Chatbot Prediction
# -------------------------------
def chatbot_reply(query):
    X = vectorizer.transform([query]).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = y_labels[predicted.item()]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I didn’t understand that."

# WhatsApp intent handler
def whatsapp_reply(query):
    result = parse_send_message_command(query)
    if not result["message"]:
        return "Sorry, I couldn't understand the WhatsApp message."
    X = w_vectorizer.transform([result["message"]]).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    output = w_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = w_y_labels[predicted.item()]
    for intent in greeting_intents["intents"]:
        if tag == intent["tag"]:
            result["message"] = intent["responses"][0].replace("99999", result["name"])
            if result["via"] == "whatsapp":
                contacts = load_contacts()
                if result["name"] in contacts:
                    send_whatsapp_message(contacts[result["name"]], result["message"])
                    return f"Message sent to {result['name']} via WhatsApp."
                else:
                    return "Contact not found."
    return "Failed to process WhatsApp message."

# -------------------------------
# MAIN LOOP
# -------------------------------
# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    wishMe()
    while True:
        query = input("Enter input:").lower()   # or query = command()

        if query == "none":
            continue
        if "exit" in query:
            speak("Goodbye Boss!")
            sys.exit()

        # ---------------- WhatsApp Handling ----------------
        if "send" in query and "via whatsapp" in query:
            response = whatsapp_reply(query)   # this uses greeting.json
            speak(response)
            print("Bot:", response)
            continue

        # ---------------- Normal Chatbot ----------------
        response = chatbot_reply(query)

        # Predict intent tag
        X = vectorizer.transform([query]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = y_labels[predicted.item()]

        # ---------------- Tag Handling ----------------
        if tag == "open_app_control":
            openApp(query)
        elif tag == "close_app_control":
            closeApp(query)
        elif "schedule" in query or "university time table" in query:
            schedule()
        elif tag == "system_conditions":
            if "volume up" in query:
                volume_control("up")
            elif "volume down" in query:
                volume_control("down")
            elif "mute" in query:
                volume_control("mute")
            elif "condition" in query:
                condition()
        else:
            speak(response)
            print("Bot:", response)

"""if __name__ == "__main__":
    wishMe()
    while True:
        query = input("Enter input:").lower()
        if query == "none":
            continue
        if "exit" in query:
            speak("Goodbye Boss!")
            sys.exit()
        elif any(x in query for x in ["facebook","discord","whatsapp","instagram"]):
            social_media(query)
        elif "schedule" in query:
            schedule()
        elif "volume up" in query:
            volume_control("up")
        elif "volume down" in query:
            volume_control("down")
        elif "open" in query:
            openApp(query)
        elif "close" in query:
            closeApp(query)
        elif "google" in query:
            browsing()
        elif "condition" in query:
            condition()
        elif "send" in query and "via whatsapp" in query:
            response = whatsapp_reply(query)
            speak(response)
            print("Bot:", response)
        else:
            response = chatbot_reply(query)
            speak(response)
            print("Bot:", response)"""
















































"""import datetime
import os
import sys
import time
import webbrowser
import pyautogui
import pyttsx3
import speech_recognition as sr
import json
import random
import numpy as np
import psutil
import subprocess
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# Tokenizer for vectorizer
# -------------------------------
def simple_tokenizer(txt):
    return txt.split()

add_safe_globals([CountVectorizer])  # safe unpickling

# -------------------------------
# Load trained chatbot model
# -------------------------------
data = torch.load("chat_model.pth", weights_only=False)
model_state = data["model_state"]
vectorizer = data["vectorizer"]
y_labels = data["y_labels"]
input_size = data["input_dim"]
hidden_size = data["hidden_dim"]
output_size = data["output_dim"]

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# -------------------------------
# Load intents
# -------------------------------
with open("data/intents.json") as file:
    intents = json.load(file)

# -------------------------------
# Text-to-Speech (Linux espeak)
# -------------------------------
engine = pyttsx3.init("espeak")   # still use "espeak" driver
engine.setProperty("voice", "english")  # espeak-ng improves voice quality
engine.setProperty("rate", 150)   # around natural speed
engine.setProperty("volume", 1.0)

def speak(text, speed=1.0, clarity=True):
    os.system(f'pico2wave -w temp.wav "{text}"')
    cmd = "sox temp.wav temp_out.wav"
    if speed != 1.0:
        cmd += f" tempo {speed}"
    if clarity:
        cmd += " treble +5 highpass 200"
    cmd += " && aplay temp_out.wav"
    os.system(cmd)

#def speak(text):
    #os.system(f'pico2wave -w temp.wav "{text}" && sox temp.wav temp_out.wav tempo 1.1 treble +5 highpass 200 && aplay temp_out.wav')
    #engine.say(text)
    #engine.runAndWait()

# -------------------------------
# Speech Recognition
# -------------------------------
def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = r.listen(source, phrase_time_limit=10)
    try:
        print("\rRecegnition...")
        query = r.recognize_google(audio, language="en-in")
        print(f"\rUser said: {query}")
    except Exception:
        print("Say that again please...")
        return "None"
    return query.lower()

# -------------------------------
# Date & Time
# -------------------------------
def cal_day():
    day = datetime.datetime.today().weekday() + 1
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][day-1]

def wishMe():
    hour = int(datetime.datetime.now().hour)
    t = time.strftime("%I:%M:%p")
    day = cal_day()
    if hour < 12:
        speak(f"Good morning Pravin, it is {day} and the time is {t}")
    elif hour < 17:
        speak(f"Good afternoon Pravin, it is {day} and the time is {t}")
    else:
        speak(f'Good evening Pravin, it is {day} and the time is {t}')

# -------------------------------
# Functions
# -------------------------------
def social_media(cmd):
    sites = {
        "facebook": "https://www.facebook.com/",
        "whatsapp": "https://web.whatsapp.com/",
        "discord": "https://discord.com/",
        "instagram": "https://www.instagram.com/"
    }
    for key, url in sites.items():
        if key in cmd:
            speak(f"Opening your {key}")
            webbrowser.open(url)
            return
    speak("No result found")

def schedule():
    today = cal_day().lower()
    week = {
    "monday": "Boss, Monday starts strong! 9–9:50 we’ve got Algorithms, then 10–11:50 System Design. Lunch at 12, and from 2 it’s Programming Lab time.",
    "tuesday": "Boss, Tuesday’s packed! 9 sharp Web Dev, 11 DB Systems. Lunch at 1, then Open Source Lab at 2.",
    "wednesday": "Boss, Wednesday vibes! 9 Machine Learning, 11 Operating Systems, 12 Ethics. Break at 1, and 2 PM Software Engineering Workshop.",
    "thursday": "Boss, Thursday flow! 9 Networks, 11 Cloud Computing. Lunch at 1, then Cybersecurity Lab from 2.",
    "friday": "Boss, it’s Friday grind! 9 Artificial Intelligence, 10 Advanced Programming, 11 UI/UX. Lunch at 1, and Capstone project work starts at 2.",
    "saturday": "Boss, Saturday schedule: morning Capstone meetings, noon Innovation session, and afternoon coding practice.",
    "sunday": "Boss, Sunday is chill mode! Take rest, maybe catch up on pending work if you like."
}
    speak(week.get(today, "No schedule found"),speed=0.8)

def openApp(cmd):
    if "calculator" in cmd:
        subprocess.Popen(["gnome-calculator"])
        speak(f"Opening calculator")
    elif "notepad" in cmd:
        subprocess.Popen(["gedit"])
        speak(f"Opening notepad")
    elif "paint" in cmd:
        subprocess.Popen(["gimp"])
        speak(f"Opening paint")

def closeApp(cmd):
    if "calculator" in cmd:
        speak(f"Closing calculator")
        os.system("pkill -f gnome-calculator")
    elif "notepad" in cmd:
        speak(f"Closing notepad")
        os.system("pkill -f gedit")
    elif "paint" in cmd:
        speak(f"Closing paint")
        os.system("pkill -f gimp")

def browsing():
    speak("Boss, what should I search on Google?")
    s = command()
    if s != "None":
        webbrowser.open(f"https://www.google.com/search?q={s}")

def condition():
    usage = str(psutil.cpu_percent())
    speak(f"CPU is at {usage} percent")
    battery = psutil.sensors_battery()
    if battery:
        speak(f"Our system has {int(battery.percent)} percent battery")
    else:
        speak("I can't detect battery on this system")

def volume_control(action):
    if action == "up":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ +5%")
        speak("Volume increased")
    elif action == "down":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ -5%")
        speak("Volume decreased")

# -------------------------------
# Chatbot Prediction
# -------------------------------
def chatbot_reply(query):
    X = vectorizer.transform([query]).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = y_labels[predicted.item()]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I didn’t understand that."

# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    wishMe()
    #speak("Allow me to introduce myself. I am Jarvis, your AI assistant on Linux.")
    while True:
        query=input("Enter input:")
        #query = command()
        if query == "None":
            continue

        if any(x in query for x in ["facebook","discord","whatsapp","instagram"]):
            social_media(query)
        elif "schedule" in query or "university time table" in query:
            schedule()
        elif "volume up" in query:
            volume_control("up")
        elif "volume down" in query:
            volume_control("down")
        elif "open" in query:
            openApp(query)
        elif "close" in query:
            closeApp(query)
        elif "google" in query:
            browsing()
        elif "condition" in query:
            condition()
        elif "exit" in query:
            speak("Goodbye Boss!")
            sys.exit()
        else:
            response = chatbot_reply(query)
            speak(response)
            print("Bot:", response)"""
