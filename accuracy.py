import json
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def simple_tokenizer(txt):
    return txt.split()

# -----------------------------
# Utils from nltk_utils.py
# -----------------------------
from nltk_utils import bag_of_words, tokenize

# -----------------------------
# Model class (from model.py)
# -----------------------------
from model import NeuralNet

# -----------------------------
# Allow CountVectorizer to unpickle safely
# -----------------------------
add_safe_globals([CountVectorizer])

# -----------------------------
# Load trained model
# -----------------------------
FILE = "nlp_model/intents.pth"   # path to your trained model
data = torch.load(FILE, weights_only=False)  # full load

input_size = data["input_dim"]
hidden_size = data["hidden_dim"]
output_size = data["output_dim"]
vectorizer = data["vectorizer"]
y_labels = data["y_labels"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# -----------------------------
# Load test data
# -----------------------------
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# -----------------------------
# Testing
# -----------------------------
total_tests = 0
correct = 0

print("\n========== TESTING START ==========\n")

for intent in test_data["intents"]:
    expected_tag = intent["tag"]
    for sentence in intent["patterns"]:
        total_tests += 1

        # Preprocess
        X = vectorizer.transform([sentence]).toarray()
        X = torch.tensor(X, dtype=torch.float32)

        # Predict
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        predicted_tag = y_labels[predicted.item()]

        # Confidence
        probs = torch.softmax(output, dim=1)
        conf = probs[0][predicted.item()].item()

        # Check pass/fail
        if predicted_tag == expected_tag:
            status = "✅ PASS"
            correct += 1
        else:
            status = "❌ FAIL"

        print(f"{status} | Input: '{sentence}' | Expected: {expected_tag} | Predicted: {predicted_tag} | Confidence: {conf:.2f}")

# -----------------------------
# Summary
# -----------------------------
accuracy = (correct / total_tests) * 100
print("\n========== TESTING SUMMARY ==========")
print(f"Total Tests   : {total_tests}")
print(f"Correct       : {correct}")
print(f"Incorrect     : {total_tests - correct}")
print(f"Accuracy      : {accuracy:.2f}%")
print("=====================================\n")
