# oversample_confused_classes.py
import shutil
from pathlib import Path

# Classes to oversample aggressively
PROBLEM_CLASSES = {
    'background_text': 10,  # Multiply by 10x
    'background': 5,        # Multiply by 5x
    'hexagon_bubble': 5,
    'wiggle_bubble': 4,
    'standalone_text': 3,
}

def oversample_by_class(labels_dir, images_dir, output_dir, multipliers):
    """Duplicate images containing specific classes"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file) as f:
            lines = f.readlines()
            classes_in_image = [int(line.split()[0]) for line in lines]
            
            # Find max multiplier for any class in this image
            max_mult = 1
            for cls_id in classes_in_image:
                if cls_id in PROBLEM_CLASSES:
                    max_mult = max(max_mult, PROBLEM_CLASSES[cls_id])
            
            # Duplicate this image max_mult times
            image_file = Path(images_dir) / f"{label_file.stem}.jpg"
            for i in range(max_mult):
                shutil.copy(image_file, output_dir / f"{label_file.stem}_dup{i}.jpg")
                shutil.copy(label_file, output_dir / f"{label_file.stem}_dup{i}.txt")