import json
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer

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
data = torch.load("chat_model.pth", weights_only=False)  # ✅ full load

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
with open("intents.json") as file:
    intents = json.load(file)

print("Chatbot is ready! Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    X = vectorizer.transform([sentence]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = y_labels[predicted.item()]

    for intent in intents['intents']:
        if tag == intent["tag"]:
            print("Bot:", intent["responses"][0])