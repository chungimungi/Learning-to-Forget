from original import TextClassifier
from original import num_epochs,num_features,y_train_tensor,X_train_tensor,X_test_tensor, criterion, y_test_tensor
from torch.optim import optim
from random_weights import randomize_weights
import torch

def create_randomized_model(input_size, output_size):
    randomized_model = TextClassifier(input_size, output_size)
    randomize_weights(randomized_model)
    return randomized_model

randomized_model = create_randomized_model(num_features, y_train_tensor.shape[1])

randomized_optimizer = optim.Adam(randomized_model.parameters(), lr=0.001)
randomized_train_losses = []
randomized_val_losses = []
randomized_val_accuracies = []
randomized_train_accuracies = []

randomized_model.train()
for epoch in range(num_epochs):
    randomized_optimizer.zero_grad()
    randomized_outputs = randomized_model(X_train_tensor)
    randomized_loss = criterion(randomized_outputs, y_train_tensor)
    randomized_loss.backward()
    randomized_optimizer.step()
    randomized_train_losses.append(randomized_loss.item())

    randomized_predicted = torch.sigmoid(randomized_outputs)
    randomized_predicted[randomized_predicted >= 0.5] = 1
    randomized_predicted[randomized_predicted < 0.5] = 0
    randomized_train_accuracy = (randomized_predicted == y_train_tensor).float().mean()
    randomized_train_accuracies.append(randomized_train_accuracy.item())

    randomized_model.eval()
    with torch.no_grad():
        randomized_val_outputs = randomized_model(X_test_tensor)
        randomized_val_loss = criterion(randomized_val_outputs, y_test_tensor)
        randomized_val_predicted = torch.sigmoid(randomized_val_outputs)
        randomized_val_predicted[randomized_val_predicted >= 0.5] = 1
        randomized_val_predicted[randomized_val_predicted < 0.5] = 0
        randomized_val_accuracy = (randomized_val_predicted == y_test_tensor).float().mean()
        randomized_val_losses.append(randomized_val_loss.item())
        randomized_val_accuracies.append(randomized_val_accuracy.item())

print(f"Epoch [{epoch+1}/{num_epochs}], \n Loss: {randomized_loss.item()}, \n Train Accuracy: {randomized_train_accuracy.item()}, \n Validation Loss: {randomized_val_loss.item()}, \n Validation Accuracy: {randomized_val_accuracy.item()}")