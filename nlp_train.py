import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import random, os

# -------------------------------
# Reproducibility
# -------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------
# Custom tokenizer
# -------------------------------
def simple_tokenizer(txt):
    return txt.split()

# -------------------------------
# Load training intents
# -------------------------------
with open("data/intents.json", "r") as f:
    intents = json.load(f)

# -------------------------------
# Preprocessing
# -------------------------------
all_patterns = []
all_tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        all_tags.append(intent["tag"])

vectorizer = CountVectorizer(tokenizer=simple_tokenizer)
X = vectorizer.fit_transform(all_patterns).toarray()

y_labels = sorted(list(set(all_tags)))
y = np.array([y_labels.index(tag) for tag in all_tags])

input_dim = X.shape[1]
hidden_dim = 8
output_dim = len(y_labels)
print(f"Input Dim: {input_dim}, Classes: {output_dim}")

# -------------------------------
# Dataset and Dataloader
# -------------------------------
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ChatDataset(X, y), batch_size=8, shuffle=True)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training
# -------------------------------
epochs = 1000
for epoch in range(epochs):
    for words, labels in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -------------------------------
# Save model
# -------------------------------
os.makedirs("nlp_model", exist_ok=True)
data = {
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "vectorizer": vectorizer,
    "y_labels": y_labels
}
torch.save(data, "nlp_model/intents.pth")

print("\nTraining complete. Model saved to 'nlp_model/intents.pth'")
