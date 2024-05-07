import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv('data/wine_review.csv')
X = data["Review_content"]
y = data["Review_name"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Determine the number of features (vocabulary size)
num_features = X_train_vectorized.shape[1]

X_train_tensor = torch.Tensor(X_train_vectorized.toarray()).to('cuda')
y_train_tensor = torch.Tensor(y_train).to('cuda')
X_test_tensor = torch.Tensor(X_test_vectorized.toarray()).to('cuda')
y_test_tensor = torch.Tensor(y_test).to('cuda')


class TextClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


initial_model = TextClassifier(num_features, y_train_tensor.shape[1]) 
optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []

num_epochs = 15
initial_model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = initial_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Compute training accuracy
    predicted = torch.sigmoid(outputs)
    predicted[predicted >= 0.5] = 1
    predicted[predicted < 0.5] = 0
    train_accuracy = (predicted == y_train_tensor).float().mean()
    train_accuracies.append(train_accuracy.item())

    # Compute validation loss and accuracy
    initial_model.eval()
    with torch.no_grad():
        val_outputs = initial_model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_predicted = torch.sigmoid(val_outputs)
        val_predicted[val_predicted >= 0.5] = 1
        val_predicted[val_predicted < 0.5] = 0
        val_accuracy = (val_predicted == y_test_tensor).float().mean()
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy.item())

print(f"Epoch [{epoch+1}/{num_epochs}], \n Loss: {loss.item()}, \n Train Accuracy: {train_accuracy.item()}, \n Validation Loss: {val_loss.item()}, \n Validation Accuracy: {val_accuracy.item()}")