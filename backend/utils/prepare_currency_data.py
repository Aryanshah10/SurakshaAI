import os, shutil, random
from pathlib import Path

SOURCE_REAL = r"C:\Users\TEST\Downloads\archive\data\data\real"   # change this
SOURCE_FAKE = r"C:\Users\TEST\Downloads\archive\data\data\fake"   # change this
DEST        = r"data/currency_dataset"

for split in ["train", "val"]:
    Path(f"{DEST}/images/{split}").mkdir(parents=True, exist_ok=True)

def split_and_copy(source_dir, label):
    images = list[Path](Path(source_dir).rglob("*.jpg")) + \
             list[Path](Path(source_dir).rglob("*.png"))
    random.shuffle(images)
    cut = int(len(images) * 0.8)
    train_imgs = images[:cut]
    val_imgs   = images[cut:]

    for img in train_imgs:
        shutil.copy(img, f"{DEST}/images/train/{label}_{img.name}")
    for img in val_imgs:
        shutil.copy(img, f"{DEST}/images/val/{label}_{img.name}")

split_and_copy(SOURCE_REAL, "real")
split_and_copy(SOURCE_FAKE, "fake")
print("Done. Images organized.")