import torch
import torch.nn as nn
import random
import json
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer

def simple_tokenizer(txt):
    return txt.split()

add_safe_globals([CountVectorizer])

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ChatbotNLP:
    def __init__(self):
        data = torch.load("nlp_model/chat_model.pth", weights_only=False)
        self.model_state = data["model_state"]
        self.vectorizer = data["vectorizer"]
        self.y_labels = data["y_labels"]
        self.input_size = data["input_dim"]
        self.hidden_size = data["hidden_dim"]
        self.output_size = data["output_dim"]

        self.model = ChatModel(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        with open("data/intents.json") as file:
            self.intents = json.load(file)

    def reply(self, query):
        X = self.vectorizer.transform([query]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.y_labels[predicted.item()]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I didn’t understand that."