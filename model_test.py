import json
import pyautogui
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer
import re
import pywhatkit
import time

# -------------------------------
# Define the tokenizer so unpickling works
# -------------------------------
def simple_tokenizer(txt):
    return txt.split()

# ✅ Allow CountVectorizer to be safely loaded
add_safe_globals([CountVectorizer])

# -----------------------------
# Load model and preprocessing objects
# -----------------------------
data = torch.load("greeting.pth", weights_only=False)  # ✅ full load

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
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load model
model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# -----------------------------
# Chat loop
# -----------------------------
with open("data/greeting.json") as file:
    intents = json.load(file)

print("Chatbot is ready! Type 'quit' to exit.")

def parse_send_message_command(cmd):
    cmd = cmd.lower()
    # Try to extract 'via' (e.g., via whatsapp, via facebook, etc.)
    via_match = re.search(r'via (\w+)', cmd)
    via = via_match.group(1) if via_match else None

    # Try to extract name (e.g., to john, to alice, to mom)
    name_match = re.search(r'to ([a-z]+)', cmd)
    name = name_match.group(1) if name_match else None

    # Try to extract message (between 'send' and 'message to' or 'to')
    msg_match = re.search(r'send (.+?) message to', cmd)
    if not msg_match:
        msg_match = re.search(r'send (.+?) to', cmd)
    message = msg_match.group(1).strip() if msg_match else None

    return {'name': name, 'message': message, 'via': via}

def load_contacts(filepath="data/contacts.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def send_whatsapp_message(phone_number, message):
    """
    Send a WhatsApp message using pywhatkit in a background thread.
    """
    pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=10, tab_close=True)
    time.sleep(4)
    pyautogui.press("enter")
    pyautogui.hotkey("ctrl", "w")  # Close the tab
    print("Bot: Message sent and tab closed.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break
    result = parse_send_message_command(sentence)
    if not result["message"]:
        print("Bot: Sorry, I couldn't understand the message to send.")
        continue
    X = vectorizer.transform([result["message"]]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = y_labels[predicted.item()]

    for intent in intents['intents']:
        if tag == intent["tag"]:
            result["message"] =intent["responses"][0].replace("99999", result["name"])
            print(f"Bot: {result}")
            if result["via"] == "whatsapp":
                contacts = load_contacts()
                if result["name"] in contacts:
                    send_whatsapp_message(contacts[result["name"]], result["message"])
                else:
                    print("Contact not found.")