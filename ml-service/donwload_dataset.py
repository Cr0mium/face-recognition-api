# download_lfw_dataset.py

import os
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np
from tqdm import tqdm

# Settings
DATA_DIR = "data"  # Inside ml-service/
MIN_FACES_PER_PERSON = 10  # Only include people with ≥10 photos
MAX_IMAGES_PER_PERSON = 20  # Optional limit for dev speed

print("📥 Downloading LFW people dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, resize=0.5)

print(f"✅ Loaded {len(lfw_people.images)} images from {len(lfw_people.target_names)} people.")

# Create output data folders
os.makedirs(DATA_DIR, exist_ok=True)

person_counts = {}
for i in tqdm(range(len(lfw_people.images)), desc="🧹 Saving images"):
    image = lfw_people.images[i]
    label = lfw_people.target[i]
    person_name = lfw_people.target_names[label]

    # Create a folder for each person
    person_folder = os.path.join(DATA_DIR, person_name.replace(" ", "_"))
    os.makedirs(person_folder, exist_ok=True)

    # Optionally limit images per person
    count = person_counts.get(person_name, 0)
    if count >= MAX_IMAGES_PER_PERSON:
        continue

    # Save image
    filename = os.path.join(person_folder, f"{count + 1}.jpg")
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(filename)
    person_counts[person_name] = count + 1

print("🎉 Dataset ready at:", DATA_DIR)
