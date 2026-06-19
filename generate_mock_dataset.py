import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from api.main import CLASS_NAMES

def create_dummy_dataset(base_dir="datasets/PlantVillage", classes=CLASS_NAMES):
    for split in ["train", "val"]:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate 2 random images per class per split for extremely fast training
            for i in range(2):
                color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
                img_array = np.full((224, 224, 3), color, dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(dir_path, f"mock_img_{i}.jpg"))
                
    print(f"Successfully generated mock dataset in {base_dir} with {len(classes)} classes")

if __name__ == "__main__":
    create_dummy_dataset()
