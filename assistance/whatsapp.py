import torch
import torch.nn as nn
import random
import json
import re
import time
import pyautogui
import pywhatkit
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer

def simple_tokenizer(txt):
    return txt.split()

add_safe_globals([CountVectorizer])

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

class WhatsAppModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WhatsAppModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class WhatsAppHandler:
    def __init__(self):
        data = torch.load("nlp_model/greeting.pth", weights_only=False)
        self.model_state = data["model_state"]
        self.vectorizer = data["vectorizer"]
        self.y_labels = data["y_labels"]
        self.input_size = data["input_dim"]
        self.hidden_size = data["hidden_dim"]
        self.output_size = data["output_dim"]

        self.model = WhatsAppModel(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        with open("data/greeting.json") as file:
            self.greeting_intents = json.load(file)

    def handle(self, query):
        result = parse_send_message_command(query)
        if not result["message"]:
            return "Sorry, I couldn't understand the WhatsApp message."
        X = self.vectorizer.transform([result["message"]]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.y_labels[predicted.item()]
        for intent in self.greeting_intents["intents"]:
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