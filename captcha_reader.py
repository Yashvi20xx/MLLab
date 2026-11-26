
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms


# 1. SELECT GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 2. RECREATE THE SAME MODEL ARCHITECTURE
class CaptchaAI(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
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

        return x.reshape(-1, 4, 10)


# 3. LOAD OUR TRAINED AI
model = CaptchaAI().to(device)

model.load_state_dict(
    torch.load(
        "results/captcha_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("AI model loaded successfully!")
print("Running on:", device)


# 4. PREPARE IMAGE
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((55, 150)),
    transforms.ToTensor()
])


# 5. PREDICT CAPTCHA
def predict_captcha(image_path):

    image = Image.open(image_path).convert("RGB")

    processed_image = transform(image)
    processed_image = processed_image.unsqueeze(0)

    processed_image = processed_image.to(device)

    with torch.no_grad():

        predictions = model(processed_image)

        probabilities = torch.softmax(
            predictions,
            dim=2
        )

        predicted_digits = predictions.argmax(dim=2)[0]

        confidence = probabilities.max(dim=2)[0][0]

    result = "".join(
        str(digit.item()) for digit in predicted_digits
    )

    scores = [
        round(value.item() * 100, 2)
        for value in confidence
    ]

    return image, result, scores


# 6. BUILD DESKTOP INTERFACE
root = tk.Tk()

root.title("My AI - CAPTCHA Reader")
root.geometry("500x520")

title = tk.Label(
    root,
    text="MY AI CAPTCHA READER",
    font=("Arial", 18, "bold")
)

title.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="Choose a CAPTCHA image",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=15)

confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 11)
)

confidence_label.pack(pady=10)

image_reference = None


def choose_image():

    global image_reference

    path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg")
        ]
    )

    if not path:
        return

    try:

        image, result, scores = predict_captcha(path)

        # Display selected image
        image.thumbnail((350, 150))

        image_reference = ImageTk.PhotoImage(image)

        image_label.config(image=image_reference)

        # Display prediction
        result_label.config(
            text=f"AI Prediction: {result}"
        )

        confidence_text = "\n".join(
            [
                f"Digit {i + 1}: {score}% confidence"
                for i, score in enumerate(scores)
            ]
        )

        confidence_label.config(text=confidence_text)

    except Exception as error:

        result_label.config(
            text=f"Error: {error}"
        )


choose_button = tk.Button(
    root,
    text="CHOOSE CAPTCHA",
    font=("Arial", 13, "bold"),
    command=choose_image
)

choose_button.pack(pady=20)


root.mainloop()
