import os
import shutil
import yaml
from pathlib import Path
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
import time
import random

class YOLODatasetBuilder:
    def __init__(self, root_folder: str, model_path: str):
        self.root_folder = Path(root_folder)
        self.model_path = model_path
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
        
    def find_all_images(self) -> List[Path]:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        image_files = []
        
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        
        return image_files
    
    def create_simple_structure(self, output_dir: str) -> Path:
        # Create output relative to script location
        script_dir = Path(__file__).parent
        output_path = script_dir / output_dir
        
        if output_path.exists():
            shutil.rmtree(output_path)
        
        folders = ['images', 'labels']
        for folder in folders:
            (output_path / folder).mkdir(parents=True, exist_ok=True)
            
        return output_path
    
    def copy_and_rename_images(self, image_files: List[Path], output_dir: Path):
        images_folder = output_dir / 'images'
        
        for idx, img_path in enumerate(image_files):
            # Sequential naming to avoid conflicts
            new_filename = f"img_{idx:06d}{img_path.suffix}"
            destination = images_folder / new_filename
            
            try:
                shutil.copy2(img_path, destination)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
                continue
    
    def auto_label_all_images(self, output_dir: Path, conf: float = 0.5, iou: float = 0.7):
        print(f"Auto-labeling with conf={conf}, iou={iou}")
        
        images_folder = output_dir / 'images'
        labels_folder = output_dir / 'labels'
        
        if not images_folder.exists():
            print("Images folder not found!")
            return
            
        image_files = list(images_folder.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in 
                      {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}]
        
        labeled_count = 0
        total_detections = 0
        total_images = len(image_files)
        start_time = time.time()
        
        for idx, img_path in enumerate(image_files):
            try:
                results = self.model(str(img_path), conf=conf, iou=iou, verbose=False)
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                height, width = img.shape[:2]
                label_path = labels_folder / f"{img_path.stem}.txt"
                
                detections_found = False
                detection_count = 0
                
                with open(label_path, 'w') as f:
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                # Fix NumPy deprecation warning
                                class_id = int(box.cls.cpu().numpy().item())
                                confidence = float(box.conf.cpu().numpy().item())
                                
                                # Convert to YOLO format (normalized coordinates)
                                x_center, y_center, w, h = box.xywh.cpu().numpy()[0]
                                x_center /= width
                                y_center /= height
                                w /= width
                                h /= height
                                
                                # Ensure coordinates are within bounds
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                w = max(0, min(1, w))
                                h = max(0, min(1, h))
                                
                                f.write(f"{class_id} {x_center:.12f} {y_center:.12f} "
                                       f"{w:.12f} {h:.12f}\n")
                                detections_found = True
                                detection_count += 1
                
                if detections_found:
                    labeled_count += 1
                    total_detections += detection_count
                
                current_time = time.time()
                elapsed_time = current_time - start_time
                progress = (idx + 1) / total_images
                
                if elapsed_time > 0:
                    speed = (idx + 1) / elapsed_time
                    eta = (total_images - idx - 1) / speed if speed > 0 else 0
                    
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    elapsed_str = f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                    eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
                    
                    progress_line = f"\r[{bar}] {idx+1}/{total_images} ({progress*100:.1f}%) | {speed:.1f} img/s | {elapsed_str} < {eta_str}"
                    print(progress_line, end='', flush=True)
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        elapsed_total = time.time() - start_time
        avg_speed = total_images / elapsed_total if elapsed_total > 0 else 0
        print(f"\nCompleted in {int(elapsed_total//60):02d}:{int(elapsed_total%60):02d}")
        print(f"Labeled {labeled_count}/{total_images} images, {total_detections} detections")
        print(f"Average speed: {avg_speed:.1f} img/s")
        
        # Preview some results
        self.preview_results(output_dir)
    
    def preview_results(self, output_dir: Path):
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
                    img_file = images_folder / f"{label_file.stem}.jpg"
                    if not img_file.exists():
                        # Try other extensions
                        for ext in ['.png', '.jpeg', '.bmp']:
                            img_file = images_folder / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                break
                    if img_file.exists():
                        labeled_files.append((img_file, label_file, lines))
        
        if not labeled_files:
            print("No labeled images found for preview")
            return
        
        # Random sample for preview
        sample_size = min(5, len(labeled_files))
        sample_files = random.sample(labeled_files, sample_size)
        
        print(f"\nCreating visual preview of {sample_size} images...")
        
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
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height
                    
                    # Convert to corner coordinates
                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)
                    
                    # Get class name and color
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    color = colors[class_id % len(colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_text = f"{class_name}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save preview image
            preview_path = preview_folder / f"preview_{idx+1}_{img_file.name}"
            cv2.imwrite(str(preview_path), img)
            print(f"  Saved: {preview_path.name} ({len(lines)} detection(s))")
        
        print(f"\nPreviews saved in: {preview_folder}")
        print("Check the preview folder to see detected bounding boxes!")
    
    def create_data_yaml(self, output_dir: Path):
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': '.',
            'val': '.', 
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"Dataset created at: {output_dir.absolute()}")
    
    def build_simple_dataset(self, output_dir: str, conf: float = 0.5, iou: float = 0.7):
        all_images = self.find_all_images()
        if not all_images:
            return None
        
        output_path = self.create_simple_structure(output_dir)
        self.copy_and_rename_images(all_images, output_path)
        self.auto_label_all_images(output_path, conf=conf, iou=iou)
        self.create_data_yaml(output_path)
        
        return output_path
    
if __name__ == "__main__":
    ROOT_FOLDER = "E:/монтаж/манга/AI Training/Raw data/Raw/Autogen dataset UPD"
    MODEL_PATH = "E:/Projects/Python/ComicGlot/models/YOLO/best.pt"
    OUTPUT_DIR = "raw_yolo_dataset"
    
    CONF = 0.3
    IOU = 0.7
    
    builder = YOLODatasetBuilder(ROOT_FOLDER, MODEL_PATH)
    builder.build_simple_dataset(OUTPUT_DIR, conf=CONF, iou=IOU)