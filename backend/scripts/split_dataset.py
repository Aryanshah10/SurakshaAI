import os, shutil, random
from pathlib import Path
    
SOURCE_REAL = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\data\data\real" # change this
SOURCE_FAKE = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\data\data\fake"  # change this
DEST        = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\Images"

for split in ["train", "val"]:
    for label in ["real", "fake"]:
        Path(f"{DEST}/{split}/{label}").mkdir(parents=True, exist_ok=True)

def split_and_copy(source_dir, label):
    images = list(Path(source_dir).rglob("*.jpg")) + \
             list(Path(source_dir).rglob("*.png")) + \
             list(Path(source_dir).rglob("*.jpeg"))

    if not images:
        print(f"Warning: No images found in {source_dir}!")
        return

    random.shuffle(images)
    cut = int(len(images) * 0.8)
    train_imgs = images[:cut]
    val_imgs   = images[cut:]

    for img in train_imgs:
        shutil.copy(img, f"{DEST}/train/{label}/{label}_{img.name}")
    for img in val_imgs:
        shutil.copy(img, f"{DEST}/val/{label}/{label}_{img.name}")

    print(f"Copied {len(train_imgs)} train and {len(val_imgs)} val images for '{label}'.")

split_and_copy(SOURCE_REAL, "real")
split_and_copy(SOURCE_FAKE, "fake")
print("Done. Images organized.")