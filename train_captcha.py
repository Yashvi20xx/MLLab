
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. CHECK GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 2. PREPARE IMAGE PROCESSING
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


# 3. LOAD OUR CAPTCHA DATASET
class CaptchaDataset(Dataset):

    def __init__(self, folder):
        self.folder = folder
        self.files = [
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        # Filename format: 7429_0.png
        label = filename.split("_")[0]

        image_path = os.path.join(self.folder, filename)

        image = Image.open(image_path)
        image = transform(image)

        # Convert each digit into a number
        target = torch.tensor(
            [int(digit) for digit in label],
            dtype=torch.long
        )

        return image, target


train_data = CaptchaDataset("captcha_data/train")
test_data = CaptchaDataset("captcha_data/test")

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_data))
print("Testing images:", len(test_data))


# 4. BUILD OUR CNN
class CaptchaAI(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 13 * 37, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 40)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        # Four positions, ten possible digits per position
        return x.reshape(-1, 4, 10)


model = CaptchaAI().to(device)


# 5. DEFINE THE LEARNING PROCESS
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 6. TRAIN THE MODEL
epochs = 8

print("\nSTARTING CAPTCHA AI TRAINING...\n")

start_time = time.time()

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_function(
            predictions.reshape(-1, 10),
            labels.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)


    # 7. TEST THE MODEL
    model.eval()

    correct_captchas = 0
    correct_digits = 0
    total_captchas = 0
    total_digits = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            predicted_digits = predictions.argmax(dim=2)

            correct_captchas += (
                (predicted_digits == labels)
                .all(dim=1)
                .sum()
                .item()
            )

            correct_digits += (
                (predicted_digits == labels)
                .sum()
                .item()
            )

            total_captchas += labels.size(0)
            total_digits += labels.numel()

    captcha_accuracy = (
        100 * correct_captchas / total_captchas
    )

    digit_accuracy = (
        100 * correct_digits / total_digits
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Digit Accuracy: {digit_accuracy:.2f}% "
        f"Complete CAPTCHA Accuracy: {captcha_accuracy:.2f}%"
    )


# 8. SAVE OUR TRAINED MODEL
os.makedirs("results", exist_ok=True)

torch.save(
    model.state_dict(),
    "results/captcha_model.pth"
)

print("\nCAPTCHA AI TRAINING COMPLETED!")

print(
    "Time taken:",
    round(time.time() - start_time, 2),
    "seconds"
)

print("Model saved: results/captcha_model.pth")
