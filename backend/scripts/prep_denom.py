
import os, shutil, random
from pathlib import Path

# Your original data source
SOURCE = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\Images"
# New destination for denomination training
DEST = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\Images_denom"

# The folders you mentioned having earlier
denominations = ["10","20", "50", "100","200","500","2000"] # Add "500_rupee" if you have it!


for split in ["train", "val"]:
    
    for denom in denominations:
        Path(f"{DEST}/{split}/{denom}").mkdir(parents=True, exist_ok=True)

for a in ['real', 'fake']:
    for denom in denominations:
        # Grab images from BOTH real and fake folders for this denomination
        images = list(Path(SOURCE).rglob(f"{denom}/*.jpg")) + \
                list(Path(SOURCE).rglob(f"{denom}/*.png")) + \
                list(Path(SOURCE).rglob(f"{denom}/*.jpeg"))
        
        random.shuffle(images)
        cut = int(len(images) * 0.8)
        
        for img in images[:cut]:
            shutil.copy(img, f"{DEST}/train/{a}/{denom}/{img.name}")
        for img in images[cut:]:
            shutil.copy(img, f"{DEST}/val/{a}/{denom}/{img.name}")

print("Done! Denomination data is ready.")