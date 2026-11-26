
import os
import random
from PIL import Image, ImageDraw, ImageFont

# 1. Create folders for our dataset
os.makedirs("captcha_data/train", exist_ok=True)
os.makedirs("captcha_data/test", exist_ok=True)

# 2. Choose a font
font_path = "C:/Windows/Fonts/arial.ttf"

if os.path.exists(font_path):
    font = ImageFont.truetype(font_path, 36)
else:
    font = ImageFont.load_default()

def generate_captcha():

    # Generate a random four-digit number
    text = "".join(
        random.choices("0123456789", k=4)
    )

    # Create a blank image
    image = Image.new("RGB", (150, 55), "white")
    draw = ImageDraw.Draw(image)

    # Draw the digits
    draw.text(
        (12, 5),
        text,
        font=font,
        fill="black"
    )

    # Add random interference lines
    for _ in range(7):
        x1 = random.randint(0, 150)
        y1 = random.randint(0, 55)
        x2 = random.randint(0, 150)
        y2 = random.randint(0, 55)

        draw.line(
            (x1, y1, x2, y2),
            fill="gray",
            width=1
        )

    # Add random noise dots
    for _ in range(100):
        x = random.randint(0, 149)
        y = random.randint(0, 54)

        draw.point((x, y), fill="gray")

    return image, text


# 3. Generate training and testing images
for split, count in [("train", 5000), ("test", 1000)]:

    print(f"Generating {split} images...")

    for i in range(count):

        image, label = generate_captcha()

        filename = f"{label}_{i}.png"

        image.save(
            f"captcha_data/{split}/{filename}"
        )

print("\nCAPTCHA DATASET CREATED!")
print("Training images: 5000")
print("Testing images: 1000")
