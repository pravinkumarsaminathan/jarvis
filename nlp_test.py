import json
import torch
import torch.nn as nn
import numpy as np
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# Tokenizer for unpickling
# -------------------------------
def simple_tokenizer(txt):
    return txt.split()

add_safe_globals([CountVectorizer])

# -------------------------------
# Load model and vectorizer
# -------------------------------
data = torch.load("nlp_model/intents.pth", weights_only=False)
model_state = data["model_state"]
vectorizer = data["vectorizer"]
y_labels = data["y_labels"]
input_dim = data["input_dim"]
hidden_dim = data["hidden_dim"]
output_dim = data["output_dim"]

# -------------------------------
# Model Definition
# -------------------------------
class ChatModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = ChatModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(model_state)
model.eval()

# -------------------------------
# Load test dataset (different file)
# -------------------------------
with open("data/test.json", "r") as f:
    test_intents = json.load(f)

# Prepare test samples
test_patterns = []
test_tags = []
for intent in test_intents["intents"]:
    for pattern in intent["patterns"]:
        test_patterns.append(pattern)
        test_tags.append(intent["tag"])

X_test = vectorizer.transform(test_patterns).toarray()
y_true = np.array([y_labels.index(tag) if tag in y_labels else -1 for tag in test_tags])
valid_mask = y_true != -1  # ignore unseen tags
X_test = X_test[valid_mask]
y_true = y_true[valid_mask]

# -------------------------------
# Evaluate
# -------------------------------
with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted.numpy() == y_true).sum()
    total = len(y_true)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

print(f"Test Accuracy (on separate file): {accuracy:.2f}% ({correct}/{total})\n")

# -------------------------------
# Chat loop
# -------------------------------
with open("data/intents.json", "r") as f:
    intents = json.load(f)

print("Chatbot is ready! Type 'quit' to exit.\n")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    X = vectorizer.transform([sentence]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = y_labels[predicted.item()]

    for intent in intents["intents"]:
        if tag == intent["tag"]:
            print("Bot:", intent["responses"][0])
            print(f"[Predicted Tag: {tag}]\n")
            break
