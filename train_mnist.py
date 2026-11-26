
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os
import time

# 1. PREPARE OUR AI LAB

device = torch.device("cuda")

print("Training device:", torch.cuda.get_device_name(0))

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. DOWNLOAD OUR DATASET

transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Batch = small group of images processed together
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_data))
print("Testing images:", len(test_data))

# 3. BUILD OUR NEURAL NETWORK

class MyAI(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)

        )

    def forward(self, x):
        return self.network(x)

model = MyAI().to(device)

# 4. DEFINE HOW AI LEARNS

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 5. TRAIN OUR AI

epochs = 5

loss_history = []
accuracy_history = []

print("\nSTARTING AI TRAINING...\n")

start_time = time.time()

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        # Move images and answers to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # AI makes predictions
        predictions = model(images)

        # Calculate how wrong it is
        loss = loss_function(predictions, labels)

        # Calculate how to improve
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # 6. TEST OUR AI

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            predicted_numbers = predictions.argmax(dim=1)

            total += labels.size(0)

            correct += (predicted_numbers == labels).sum().item()

    accuracy = 100 * correct / total

    loss_history.append(average_loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# 7. SAVE OUR TRAINED AI

torch.save(model.state_dict(), "results/mnist_model.pth")

# 8. CREATE TRAINING GRAPHS

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(loss_history, marker="o")

plt.title("AI Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.subplot(1, 2, 2)

plt.plot(accuracy_history, marker="o")

plt.title("AI Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()

plt.tight_layout()

plt.savefig("results/training_graphs.png")

plt.show()

print("\nTRAINING COMPLETED!")
print("Final accuracy:", round(accuracy_history[-1], 2), "%")
print("Time taken:", round(time.time() - start_time, 2), "seconds")

print("\nYour trained model is saved!")
print("Your graphs are saved!")
