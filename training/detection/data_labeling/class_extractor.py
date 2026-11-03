import os
import shutil
import yaml
from pathlib import Path
from typing import List, Set
import cv2
import numpy as np
from ultralytics import YOLO
import time
import random
import hashlib

class FilteredYOLODatasetBuilder:
    def __init__(self, root_folder: str, model_path: str, target_classes: List[str]):
        self.root_folder = Path(root_folder)
        self.model_path = model_path
        self.target_classes = set(target_classes)
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.class_names = [
            'background_text', 'cloud_bubble', 'hexagon_bubble', 'scream_bubble', 
            'sfx_sound', 'speech_bubble', 'spiked_bubble', 'sqare_bubble', 
            'standalone_text', 'system_bubble', 'think_bubble', 'title_panel', 
            'wiggle_bubble'
        ]

        self.class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.target_class_indices = set()
        for class_name in self.target_classes:
            if class_name in self.class_name_to_idx:
                self.target_class_indices.add(self.class_name_to_idx[class_name])
            else:
                print(f"Warning: Class '{class_name}' not found in model classes")
        
        print(f"Target classes: {list(self.target_classes)}")
        print(f"Target class indices: {list(self.target_class_indices)}")
        
    def find_all_images(self) -> List[Path]:
        """Recursively find all images in all subdirectories"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        image_files = []
        
        print(f"Scanning directory: {self.root_folder}")
        
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        
        print(f"Found {len(image_files)} images total")
        return image_files
    
    def create_unique_filename(self, original_path: Path) -> str:
        """Create a unique filename preserving the directory structure"""
        try:
            relative_path = original_path.relative_to(self.root_folder)
        except ValueError:
            relative_path = original_path

        path_str = str(relative_path.parent)
        if path_str == '.':
            path_str = ''
        else:
            path_str = path_str.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
            path_str = path_str + '_'

        unique_name = f"{path_str}{original_path.stem}{original_path.suffix}"

        if len(unique_name) > 200 or any(char in unique_name for char in '<>:"/\\|?*'):
            hash_part = hashlib.md5(str(relative_path).encode()).hexdigest()[:8]
            unique_name = f"{hash_part}_{original_path.name}"
        
        return unique_name
    
    def create_simple_structure(self, output_dir: str) -> Path:
        """Create output directory structure"""
        script_dir = Path(__file__).parent
        output_path = script_dir / output_dir
        
        if output_path.exists():
            shutil.rmtree(output_path)
        
        folders = ['images', 'labels']
        for folder in folders:
            (output_path / folder).mkdir(parents=True, exist_ok=True)
            
        return output_path
    
    def has_target_class(self, img_path: Path, conf: float = 0.5, iou: float = 0.7) -> tuple:
        """Check if image contains target class and return detections"""
        try:
            results = self.model(str(img_path), conf=conf, iou=iou, verbose=False)
            
            target_detections = []
            all_detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls.cpu().numpy().item())
                        confidence = float(box.conf.cpu().numpy().item())

                        detection_data = {
                            'class_id': class_id,
                            'confidence': confidence,
                            'box': box
                        }
                        all_detections.append(detection_data)

                        if class_id in self.target_class_indices:
                            target_detections.append(detection_data)
            
            has_target = len(target_detections) > 0
            return has_target, all_detections, target_detections
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False, [], []
    
    def save_detections_to_label(self, detections: List[dict], img_path: Path, 
                                label_path: Path, img_width: int, img_height: int):
        """Save detections to YOLO format label file"""
        with open(label_path, 'w') as f:
            for detection in detections:
                box = detection['box']
                class_id = detection['class_id']

                x_center, y_center, w, h = box.xywh.cpu().numpy()[0]
                x_center /= img_width
                y_center /= img_height
                w /= img_width
                h /= img_height

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                f.write(f"{class_id} {x_center:.12f} {y_center:.12f} "
                       f"{w:.12f} {h:.12f}\n")
    
    def build_filtered_dataset(self, output_dir: str, conf: float = 0.5, iou: float = 0.7):
        """Build dataset containing only images with target classes"""
        print(f"Building filtered dataset for classes: {list(self.target_classes)}")
        print(f"Using conf={conf}, iou={iou}")

        all_images = self.find_all_images()
        if not all_images:
            print("No images found!")
            return None

        output_path = self.create_simple_structure(output_dir)
        images_folder = output_path / 'images'
        labels_folder = output_path / 'labels'

        filtered_count = 0
        total_target_detections = 0
        total_all_detections = 0
        start_time = time.time()
        
        print(f"\nProcessing {len(all_images)} images...")
        
        for idx, img_path in enumerate(all_images):
            try:

                has_target, all_detections, target_detections = self.has_target_class(
                    img_path, conf=conf, iou=iou)
                
                if has_target:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"\nCannot read image: {img_path}")
                        continue
                    
                    height, width = img.shape[:2]

                    unique_filename = self.create_unique_filename(img_path)
                    img_destination = images_folder / unique_filename
                    try:
                        shutil.copy2(img_path, img_destination)
                    except Exception as e:
                        print(f"\nError copying {img_path}: {e}")
                        continue

                    label_filename = Path(unique_filename).stem + '.txt'
                    label_destination = labels_folder / label_filename

                    self.save_detections_to_label(all_detections, img_path, 
                                                label_destination, width, height)
                    
                    filtered_count += 1
                    total_target_detections += len(target_detections)
                    total_all_detections += len(all_detections)

                    if filtered_count <= 5:
                        target_classes_found = [self.class_names[d['class_id']] for d in target_detections]
                        print(f"\nFound image {filtered_count}: {img_path.name}")
                        print(f"  Target classes: {target_classes_found}")
                        print(f"  Total detections: {len(all_detections)}")
                        print(f"  Saved as: {unique_filename}")

                current_time = time.time()
                elapsed_time = current_time - start_time
                progress = (idx + 1) / len(all_images)
                
                if elapsed_time > 0:
                    speed = (idx + 1) / elapsed_time
                    eta = (len(all_images) - idx - 1) / speed if speed > 0 else 0
                    
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    elapsed_str = f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                    eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
                    
                    progress_line = (f"\r[{bar}] {idx+1}/{len(all_images)} ({progress*100:.1f}%) | "
                                   f"Found: {filtered_count} | {speed:.1f} img/s | {elapsed_str} < {eta_str}")
                    print(progress_line, end='', flush=True)
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        elapsed_total = time.time() - start_time
        avg_speed = len(all_images) / elapsed_total if elapsed_total > 0 else 0
        
        print(f"\n\nCompleted in {int(elapsed_total//60):02d}:{int(elapsed_total%60):02d}")
        print(f"Filtered {filtered_count}/{len(all_images)} images ({filtered_count/len(all_images)*100:.1f}%)")
        print(f"Target class detections: {total_target_detections}")
        print(f"Total detections saved: {total_all_detections}")
        print(f"Average speed: {avg_speed:.1f} img/s")
        
        if filtered_count > 0:
            # Create data.yaml
            self.create_data_yaml(output_path)
            
            # Create preview
            self.preview_results(output_path)
            
            print(f"\nFiltered dataset created at: {output_path.absolute()}")
            return output_path
        else:
            print(f"\nNo images found containing target classes: {list(self.target_classes)}")
            return None
    
    def preview_results(self, output_dir: Path):
        """Create visual preview of some filtered results"""
        images_folder = output_dir / 'images'
        labels_folder = output_dir / 'labels'
        preview_folder = output_dir / 'preview'
        
        # Create preview folder
        preview_folder.mkdir(exist_ok=True)
        
        # Get labeled images
        label_files = list(labels_folder.glob('*.txt'))
        labeled_files = []
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    img_file = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']:
                        potential_img = images_folder / f"{label_file.stem}{ext}"
                        if potential_img.exists():
                            img_file = potential_img
                            break
                    
                    if img_file:
                        labeled_files.append((img_file, label_file, lines))
        
        if not labeled_files:
            print("No labeled images found for preview")
            return
        
        # Random sample for preview
        sample_size = min(5, len(labeled_files))
        sample_files = random.sample(labeled_files, sample_size)
        
        print(f"\nCreating visual preview of {sample_size} images...")
        
        # Colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
            (255, 192, 203), (0, 100, 0), (139, 69, 19)
        ]
        
        for idx, (img_file, label_file, lines) in enumerate(sample_files):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            height, width = img.shape[:2]
            target_detections = 0
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height

                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)

                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    color = colors[class_id % len(colors)]
                    
                    is_target = class_id in self.target_class_indices
                    if is_target:
                        target_detections += 1
                        thickness = 4
                        class_name = f"🎯 {class_name}"
                    else:
                        thickness = 2

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            preview_path = preview_folder / f"preview_{idx+1}_{img_file.name}"
            cv2.imwrite(str(preview_path), img)
            print(f"  Saved: {preview_path.name} ({len(lines)} total, {target_detections} target)")
        
        print(f"\nPreviews saved in: {preview_folder}")
    
    def create_data_yaml(self, output_dir: Path):
        """Create data.yaml file for the dataset"""
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': '.',
            'val': '.', 
            'nc': len(self.class_names),
            'names': self.class_names,
            'target_classes': list(self.target_classes)
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"Data configuration saved: {yaml_path}")

# Example usage
if __name__ == "__main__":
    ROOT_FOLDER = "e:/монтаж/манга/AI Training/Raw data/google script/downloads 2 [stitched]"
    MODEL_PATH = "E:/Projects/Python/ComicGlot/models/YOLO/best.pt"
    OUTPUT_DIR = "magician_bubble_dataset"

    TARGET_CLASSES = ["cloud_bubble", "hexagon_bubble", "title_panel"]

    CONF = 0.3
    IOU = 0.7
    
    builder = FilteredYOLODatasetBuilder(ROOT_FOLDER, MODEL_PATH, TARGET_CLASSES)
    result = builder.build_filtered_dataset(OUTPUT_DIR, conf=CONF, iou=IOU)