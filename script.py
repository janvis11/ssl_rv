import random
import shutil
from pathlib import Path

source_dir = Path("")
target_dir = Path("")
num_images = 500
seed = 42

random.seed(seed)
target_dir.mkdir(parents=True, exist_ok=True)

images = sorted(source_dir.glob("*.png"))
sampled = random.sample(images, min(num_images, len(images)))

for img in sampled:
    shutil.copy2(img, target_dir / img.name)

print(f"Copied {len(sampled)} images to {target_dir}")
