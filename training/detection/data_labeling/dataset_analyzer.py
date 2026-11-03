import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from collections import defaultdict
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Set
import argparse
from matplotlib.patches import Rectangle
import random

class DatasetAnalyzer:
    def __init__(self, dataset_path: str, model_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path
        self.model = None
        if model_path:
            self.model = YOLO(model_path)

        self.class_names = [
            'background_text', 'cloud_bubble', 'hexagon_bubble', 'scream_bubble', 
            'sfx_sound', 'speech_bubble', 'spiked_bubble', 'sqare_bubble', 
            'standalone_text', 'system_bubble', 'think_bubble', 'title_panel', 
            'wiggle_bubble'
        ]
        
        self.splits = ['train', 'valid']
    
    def parse_polygon_label(self, line: str) -> Tuple[int, List[float]]:
        """Parse polygon format label and convert to YOLO bbox format."""
        parts = line.strip().split()
        
        if len(parts) >= 9:  # Polygon format: class_id + 8+ coordinates
            class_id = int(parts[0])
            # Get all coordinate pairs (skip the class_id)
            coords = [float(x) for x in parts[1:]]
            
            # Extract x and y coordinates
            x_coords = coords[0::2]  # Every even index: x coordinates
            y_coords = coords[1::2]  # Every odd index: y coordinates
            
            # Calculate bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Convert to YOLO format (center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            return class_id, [x_center, y_center, width, height]
            
        elif len(parts) >= 5:  # Standard YOLO format
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            return class_id, bbox
        
        return None, None
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union of two YOLO format bounding boxes."""
        # Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)
        def yolo_to_xyxy(box):
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = yolo_to_xyxy(box1[1:5])  # Skip class_id
        box2_xyxy = yolo_to_xyxy(box2[1:5])  # Skip class_id

        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)

        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def calculate_iou_simple(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU for two bboxes in YOLO format [x_center, y_center, width, height]."""
        # Convert YOLO format to [x1, y1, x2, y2]
        def yolo_to_xyxy(bbox):
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = yolo_to_xyxy(bbox1)
        box2_xyxy = yolo_to_xyxy(bbox2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_duplicate_detections(self, iou_threshold: float = 0.9) -> Dict:
        """Find images with duplicate detections (high IoU but different classes)."""
        duplicates = {}
        
        for split in self.splits:
            labels_dir = self.dataset_path / split / 'labels'
            if not labels_dir.exists():
                print(f"Warning: {labels_dir} does not exist, skipping {split}")
                continue
                
            split_duplicates = []
            
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                annotations = []
                for line in lines:
                    class_id, bbox = self.parse_polygon_label(line)
                    if class_id is not None and bbox is not None:
                        annotations.append([class_id] + bbox)

                file_duplicates = []
                for i in range(len(annotations)):
                    for j in range(i + 1, len(annotations)):
                        ann1, ann2 = annotations[i], annotations[j]
                        
                        if ann1[0] != ann2[0]:
                            iou = self.calculate_iou(ann1, ann2)
                            if iou >= iou_threshold:
                                file_duplicates.append({
                                    'ann1': {'class': ann1[0], 'class_name': self.class_names[ann1[0]], 'bbox': ann1[1:5]},
                                    'ann2': {'class': ann2[0], 'class_name': self.class_names[ann2[0]], 'bbox': ann2[1:5]},
                                    'iou': iou
                                })
                
                if file_duplicates:
                    split_duplicates.append({
                        'file': label_file.name,
                        'image_file': label_file.name.replace('.txt', '.jpg'),
                        'duplicates': file_duplicates
                    })
            
            if split_duplicates:
                duplicates[split] = split_duplicates
        
        return duplicates
    
    def save_duplicate_report(self, duplicates: Dict, output_dir: str = 'duplicate_analysis'):
        """Save duplicate detection report and copy problematic images."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report = []
        for split, split_duplicates in duplicates.items():
            for item in split_duplicates:
                for dup in item['duplicates']:
                    report.append({
                        'split': split,
                        'file': item['file'],
                        'image_file': item['image_file'],
                        'class1': dup['ann1']['class_name'],
                        'class2': dup['ann2']['class_name'],
                        'iou': dup['iou']
                    })

        df = pd.DataFrame(report)
        df.to_csv(output_path / 'duplicate_detections_report.csv', index=False)
        
        problem_images_dir = output_path / 'problem_images'
        problem_images_dir.mkdir(exist_ok=True)
        
        for split, split_duplicates in duplicates.items():
            split_dir = problem_images_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for item in split_duplicates:
                # Copy image
                src_img = self.dataset_path / split / 'images' / item['image_file']
                if src_img.exists():
                    shutil.copy2(src_img, split_dir / item['image_file'])
                
                # Copy label
                src_label = self.dataset_path / split / 'labels' / item['file']
                if src_label.exists():
                    shutil.copy2(src_label, split_dir / item['file'])
        
        print(f"Duplicate analysis saved to {output_path}")
        print(f"Found {len(df)} duplicate detections across {len(df.groupby(['split', 'file']))} files")
        
        return df
    
    def run_model_validation(self, confidence_threshold: float = 0.25, 
                           iou_threshold: float = 0.5, save_misdetections: bool = True,
                           visualize: bool = True, sample_percentage: float = 100.0) -> Dict:
        """Run model inference on dataset and find misdetections."""
        if not self.model:
            raise ValueError("Model not loaded. Please provide model_path when initializing.")
        
        misdetections = {}
        
        for split in self.splits:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: {split} directories don't exist, skipping")
                continue
            
            split_misdetections = []
            
            for img_file in images_dir.glob('*.jpg'):
                label_file = labels_dir / (img_file.stem + '.txt')
                
                if not label_file.exists():
                    continue

                gt_annotations = []
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id, bbox = self.parse_polygon_label(line)
                        if class_id is not None and bbox is not None:
                            gt_annotations.append({'class': class_id, 'bbox': bbox})

                results = self.model(str(img_file), conf=confidence_threshold, iou=iou_threshold)

                predictions = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Convert to YOLO format
                        xyxy = box.xyxy[0].cpu().numpy()
                        img = cv2.imread(str(img_file))
                        h, w = img.shape[:2]
                        
                        x_center = (xyxy[0] + xyxy[2]) / 2 / w
                        y_center = (xyxy[1] + xyxy[3]) / 2 / h
                        width = (xyxy[2] - xyxy[0]) / w
                        height = (xyxy[3] - xyxy[1]) / h
                        
                        predictions.append({
                            'class': int(box.cls.cpu().numpy()[0]),
                            'confidence': float(box.conf.cpu().numpy()[0]),
                            'bbox': [x_center, y_center, width, height]
                        })

                is_misdetection = self.analyze_predictions(gt_annotations, predictions)
                
                if is_misdetection:
                    split_misdetections.append({
                        'file': img_file.name,
                        'image_path': str(img_file),
                        'ground_truth': gt_annotations,
                        'predictions': predictions,
                        'analysis': is_misdetection
                    })
            
            if split_misdetections:
                misdetections[split] = split_misdetections
        
        if save_misdetections:
            self.save_misdetection_report(misdetections)
        
        if visualize and misdetections:
            self.visualize_misdetections(misdetections, sample_percentage=sample_percentage)
        
        return misdetections
    
    def analyze_predictions(self, gt_annotations: List[Dict], predictions: List[Dict]) -> Dict:
        """Analyze if predictions match ground truth."""
        analysis = {
            'missing_detections': [],
            'false_positives': [],
            'wrong_classifications': []
        }
        
        used_predictions = set()
        
        for gt_idx, gt in enumerate(gt_annotations):
            best_match = None
            best_iou = 0
            best_pred_idx = -1
            
            for pred_idx, pred in enumerate(predictions):
                if pred_idx in used_predictions:
                    continue
                
                # Calculate IoU between ground truth and prediction
                iou = self.calculate_iou_simple(gt['bbox'], pred['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred
                    best_pred_idx = pred_idx
            
            if best_iou < 0.5:  # No good match found
                analysis['missing_detections'].append(gt)
            elif best_match['class'] != gt['class']:  # Wrong classification
                analysis['wrong_classifications'].append({
                    'ground_truth': gt,
                    'prediction': best_match,
                    'iou': best_iou
                })
                used_predictions.add(best_pred_idx)
            else:
                # Correct detection!
                used_predictions.add(best_pred_idx)
        
        # Check for false positives
        for pred_idx, pred in enumerate(predictions):
            if pred_idx not in used_predictions:
                analysis['false_positives'].append(pred)
        
        # Return analysis if there are any issues
        if (analysis['missing_detections'] or 
            analysis['false_positives'] or 
            analysis['wrong_classifications']):
            return analysis
        
        return None
    
    def visualize_misdetections(self, misdetections: Dict, output_dir: str = 'misdetection_analysis', 
                              sample_percentage: float = 100.0):
        """Create comprehensive visualizations of misdetection data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Prepare data for analysis
        all_data = []
        confidence_data = []
        confusion_pairs = []
        
        for split, split_misdetections in misdetections.items():
            for item in split_misdetections:
                analysis = item['analysis']
                
                # Collect data for various analyses
                for missing in analysis['missing_detections']:
                    all_data.append({
                        'split': split,
                        'file': item['file'],
                        'error_type': 'Missing Detection',
                        'gt_class': missing['class'],
                        'gt_class_name': self.class_names[missing['class']],
                        'pred_class': None,
                        'pred_class_name': None,
                        'confidence': None,
                        'iou': None
                    })
                
                for fp in analysis['false_positives']:
                    all_data.append({
                        'split': split,
                        'file': item['file'],
                        'error_type': 'False Positive',
                        'gt_class': None,
                        'gt_class_name': None,
                        'pred_class': fp['class'],
                        'pred_class_name': self.class_names[fp['class']],
                        'confidence': fp['confidence'],
                        'iou': None
                    })
                    confidence_data.append({
                        'error_type': 'False Positive',
                        'class_name': self.class_names[fp['class']],
                        'confidence': fp['confidence']
                    })
                
                for wrong in analysis['wrong_classifications']:
                    gt_class = wrong['ground_truth']['class']
                    pred_class = wrong['prediction']['class']
                    all_data.append({
                        'split': split,
                        'file': item['file'],
                        'error_type': 'Wrong Classification',
                        'gt_class': gt_class,
                        'gt_class_name': self.class_names[gt_class],
                        'pred_class': pred_class,
                        'pred_class_name': self.class_names[pred_class],
                        'confidence': wrong['prediction']['confidence'],
                        'iou': wrong['iou']
                    })
                    confidence_data.append({
                        'error_type': 'Wrong Classification',
                        'class_name': self.class_names[pred_class],
                        'confidence': wrong['prediction']['confidence']
                    })
                    confusion_pairs.append({
                        'true_class': self.class_names[gt_class],
                        'predicted_class': self.class_names[pred_class],
                        'iou': wrong['iou']
                    })
        
        df_all = pd.DataFrame(all_data)
        df_confidence = pd.DataFrame(confidence_data)
        df_confusion = pd.DataFrame(confusion_pairs)
        
        # Create visualizations
        self._plot_error_distribution(df_all, output_path)
        self._plot_class_wise_errors(df_all, output_path)
        self._plot_confidence_distributions(df_confidence, output_path)
        if not df_confusion.empty:
            self._plot_confusion_matrix(df_confusion, output_path)
            self._plot_iou_distributions(df_confusion, output_path)
        self._save_annotated_images(misdetections, output_path, sample_percentage=sample_percentage)
        
        print(f"Visualizations saved to {output_path}")
    
    def _plot_error_distribution(self, df: pd.DataFrame, output_path: Path):
        """Plot distribution of error types."""
        plt.figure(figsize=(12, 8))
        
        # Overall error distribution
        plt.subplot(2, 2, 1)
        error_counts = df['error_type'].value_counts()
        plt.bar(error_counts.index, error_counts.values, alpha=0.7)
        plt.title('Overall Error Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # Error distribution by split
        plt.subplot(2, 2, 2)
        split_error = df.groupby(['split', 'error_type']).size().unstack(fill_value=0)
        split_error.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.title('Error Distribution by Split')
        plt.xticks(rotation=0)
        plt.ylabel('Count')
        plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Files with errors by split
        plt.subplot(2, 2, 3)
        files_with_errors = df.groupby('split')['file'].nunique()
        plt.bar(files_with_errors.index, files_with_errors.values, alpha=0.7)
        plt.title('Files with Errors by Split')
        plt.ylabel('Number of Files')
        
        # Average errors per file
        plt.subplot(2, 2, 4)
        avg_errors = df.groupby('split').size() / df.groupby('split')['file'].nunique()
        plt.bar(avg_errors.index, avg_errors.values, alpha=0.7)
        plt.title('Average Errors per File')
        plt.ylabel('Errors per File')
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_class_wise_errors(self, df: pd.DataFrame, output_path: Path):
        """Plot errors by class."""
        plt.figure(figsize=(15, 10))
        
        # Missing detections by class
        plt.subplot(2, 2, 1)
        missing = df[df['error_type'] == 'Missing Detection']
        if not missing.empty:
            missing_counts = missing['gt_class_name'].value_counts()
            plt.barh(missing_counts.index, missing_counts.values, alpha=0.7, color='red')
            plt.title('Missing Detections by Class')
            plt.xlabel('Count')
        
        # False positives by class
        plt.subplot(2, 2, 2)
        fp = df[df['error_type'] == 'False Positive']
        if not fp.empty:
            fp_counts = fp['pred_class_name'].value_counts()
            plt.barh(fp_counts.index, fp_counts.values, alpha=0.7, color='orange')
            plt.title('False Positives by Class')
            plt.xlabel('Count')
        
        # Wrong classifications - true classes
        plt.subplot(2, 2, 3)
        wrong = df[df['error_type'] == 'Wrong Classification']
        if not wrong.empty:
            wrong_gt = wrong['gt_class_name'].value_counts()
            plt.barh(wrong_gt.index, wrong_gt.values, alpha=0.7, color='blue')
            plt.title('Wrong Classifications - True Classes')
            plt.xlabel('Count')
        
        # Wrong classifications - predicted classes
        plt.subplot(2, 2, 4)
        if not wrong.empty:
            wrong_pred = wrong['pred_class_name'].value_counts()
            plt.barh(wrong_pred.index, wrong_pred.values, alpha=0.7, color='purple')
            plt.title('Wrong Classifications - Predicted Classes')
            plt.xlabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_path / 'class_wise_errors.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_distributions(self, df: pd.DataFrame, output_path: Path):
        """Plot confidence score distributions for different error types."""
        if df.empty:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Confidence distribution by error type
        plt.subplot(1, 3, 1)
        for error_type in df['error_type'].unique():
            data = df[df['error_type'] == error_type]['confidence']
            plt.hist(data, alpha=0.6, label=error_type, bins=20)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Error Type')
        plt.legend()
        
        # Box plot of confidence by error type
        plt.subplot(1, 3, 2)
        error_types = df['error_type'].unique()
        conf_data = [df[df['error_type'] == et]['confidence'].dropna() for et in error_types]
        plt.boxplot(conf_data, labels=error_types)
        plt.xticks(rotation=45)
        plt.ylabel('Confidence Score')
        plt.title('Confidence Distribution Box Plot')
        
        # Confidence by class for false positives
        plt.subplot(1, 3, 3)
        fp_data = df[df['error_type'] == 'False Positive']
        if not fp_data.empty:
            fp_by_class = fp_data.groupby('class_name')['confidence'].mean().sort_values()
            plt.barh(fp_by_class.index, fp_by_class.values, alpha=0.7)
            plt.xlabel('Average Confidence')
            plt.title('Average False Positive Confidence by Class')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrix(self, df: pd.DataFrame, output_path: Path):
        """Plot confusion matrix for wrong classifications."""
        if df.empty:
            return
            
        # Create confusion matrix
        confusion_counts = df.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Reds', 
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix for Wrong Classifications')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_iou_distributions(self, df: pd.DataFrame, output_path: Path):
        """Plot IoU distributions for wrong classifications."""
        if df.empty:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Overall IoU distribution
        plt.subplot(1, 2, 1)
        plt.hist(df['iou'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title('IoU Distribution for Wrong Classifications')
        plt.axvline(df['iou'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["iou"].mean():.3f}')
        plt.legend()
        
        # IoU by confusion pair (top 10)
        plt.subplot(1, 2, 2)
        df['confusion_pair'] = df['true_class'] + ' → ' + df['predicted_class']
        top_pairs = df['confusion_pair'].value_counts().head(10).index
        top_data = df[df['confusion_pair'].isin(top_pairs)]
        
        if not top_data.empty:
            confusion_iou = top_data.groupby('confusion_pair')['iou'].mean().sort_values()
            plt.barh(range(len(confusion_iou)), confusion_iou.values, alpha=0.7)
            plt.yticks(range(len(confusion_iou)), confusion_iou.index)
            plt.xlabel('Average IoU')
            plt.title('Average IoU for Top Confusion Pairs')
        
        plt.tight_layout()
        plt.savefig(output_path / 'iou_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_annotated_images(self, misdetections: Dict, output_path: Path, sample_percentage: float = 100.0):
        """Save annotated images with ground truth and predictions overlaid."""
        # Collect all samples
        all_samples = []
        for split, split_misdetections in misdetections.items():
            for item in split_misdetections:
                all_samples.append((split, item))
        
        if not all_samples:
            return
        
        # Calculate number of samples based on percentage
        total_samples = len(all_samples)
        num_samples = int(total_samples * sample_percentage / 100.0)
        num_samples = min(num_samples, total_samples)  # Don't exceed available samples
        
        # Select samples (you can randomize this if needed)
        if sample_percentage < 100.0:
            # Random sampling
            random.seed(42)  # For reproducible results
            samples = random.sample(all_samples, num_samples)
        else:
            samples = all_samples
        
        print(f"Saving {num_samples} out of {total_samples} annotated images ({sample_percentage}%)")
        
        if num_samples == 0:
            print("No samples to save.")
            return
        
        # Create output directory for annotated images
        annotated_dir = output_path / 'annotated_misdetections'
        annotated_dir.mkdir(exist_ok=True)
        
        # Define colors (BGR format for OpenCV)
        colors = {
            'ground_truth': (0, 255, 0),      # Green
            'false_positive': (0, 0, 255),    # Red
            'wrong_classification': (0, 165, 255),  # Orange
            'missing_detection': (255, 0, 0)   # Blue
        }
        
        for idx, (split, item) in enumerate(samples):
            # Load image
            img_path = item['image_path']
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            h, w = img.shape[:2]
            analysis = item['analysis']
            
            # Create a copy for annotation
            annotated_img = img.copy()
            
            # Draw ground truth boxes
            for gt_idx, gt in enumerate(item['ground_truth']):
                bbox = gt['bbox']
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), colors['ground_truth'], 2)
                
                # Add class label
                class_name = self.class_names[gt['class']]
                label = f"GT: {class_name}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_img, 
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width + 5, y1),
                            colors['ground_truth'], -1)
                
                # Draw text
                cv2.putText(annotated_img, label, (x1 + 2, y1 - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw predictions
            for pred_idx, pred in enumerate(item['predictions']):
                bbox = pred['bbox']
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Determine if this is a false positive or wrong classification
                is_fp = any(fp == pred for fp in analysis['false_positives'])
                is_wrong = any(wrong['prediction'] == pred for wrong in analysis['wrong_classifications'])
                
                color = colors['false_positive'] if is_fp else colors['wrong_classification']
                error_type = "FP" if is_fp else "WRONG"
                
                # Draw rectangle (dashed effect with multiple thin rectangles)
                for offset in range(0, 3):
                    cv2.rectangle(annotated_img, 
                                (x1 + offset, y1 + offset), 
                                (x2 - offset, y2 - offset), 
                                color, 1)
                
                # Add prediction label with confidence
                class_name = self.class_names[pred['class']]
                confidence = pred['confidence']
                label = f"{error_type}: {class_name} ({confidence:.2f})"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Position text above the box, but offset if ground truth text is there
                text_y = y1 - baseline - 2
                if any(gt['bbox'][1] * h <= y1 + 30 for gt in item['ground_truth']):
                    text_y = y2 + text_height + baseline + 5
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_img, 
                            (x1, text_y - text_height - baseline - 5),
                            (x1 + text_width + 5, text_y + baseline),
                            color, -1)
                
                # Draw text
                cv2.putText(annotated_img, label, (x1 + 2, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add error summary in top-left corner
            error_summary = []
            if analysis['missing_detections']:
                error_summary.append(f"Missing: {len(analysis['missing_detections'])}")
            if analysis['false_positives']:
                error_summary.append(f"FP: {len(analysis['false_positives'])}")
            if analysis['wrong_classifications']:
                error_summary.append(f"Wrong: {len(analysis['wrong_classifications'])}")
            
            if error_summary:
                summary_text = " | ".join(error_summary)
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw background rectangle
                cv2.rectangle(annotated_img, 
                            (10, 10),
                            (20 + text_width, 30 + text_height),
                            (0, 255, 255), -1)  # Yellow background
                
                # Draw text
                cv2.putText(annotated_img, summary_text, (15, 25 + text_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Create filename with error information
            base_name = Path(item['file']).stem
            error_info = []
            if analysis['missing_detections']:
                error_info.append(f"miss{len(analysis['missing_detections'])}")
            if analysis['false_positives']:
                error_info.append(f"fp{len(analysis['false_positives'])}")
            if analysis['wrong_classifications']:
                error_info.append(f"wrong{len(analysis['wrong_classifications'])}")
            
            error_suffix = "_" + "_".join(error_info) if error_info else ""
            filename = f"{split}_{base_name}{error_suffix}_annotated.jpg"
            
            # Save annotated image
            output_file = annotated_dir / filename
            cv2.imwrite(str(output_file), annotated_img)
        
        print(f"Saved {len(samples)} annotated images to {annotated_dir}")
        
        # Create a summary text file
        summary_file = annotated_dir / 'annotation_legend.txt'
        with open(summary_file, 'w') as f:
            f.write("ANNOTATION LEGEND\n")
            f.write("================\n\n")
            f.write("Ground Truth Boxes:\n")
            f.write("- Color: Green solid rectangles\n")
            f.write("- Label: GT: <class_name>\n\n")
            f.write("Prediction Boxes:\n")
            f.write("- False Positives: Red dashed rectangles\n")
            f.write("  Label: FP: <class_name> (<confidence>)\n")
            f.write("- Wrong Classifications: Orange dashed rectangles\n")
            f.write("  Label: WRONG: <class_name> (<confidence>)\n\n")
            f.write("Error Summary (top-left corner):\n")
            f.write("- Yellow box with error counts\n")
            f.write("- Format: Missing: X | FP: Y | Wrong: Z\n\n")
            f.write("Filename Format:\n")
            f.write("- <split>_<original_name>_<error_info>_annotated.jpg\n")
            f.write("- Error info examples: miss2_fp1, wrong3, fp2_wrong1\n")
        
        print(f"Created annotation legend at {summary_file}")
    
    def save_misdetection_report(self, misdetections: Dict, output_dir: str = 'misdetection_analysis'):
        """Save misdetection report and copy problematic images."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_data = []
        
        for split, split_misdetections in misdetections.items():
            split_dir = output_path / 'misdetected_images' / split
            split_dir.mkdir(exist_ok=True, parents=True)
            
            for item in split_misdetections:
                analysis = item['analysis']

                src_img = self.dataset_path / split / 'images' / item['file']
                if src_img.exists():
                    shutil.copy2(src_img, split_dir / item['file'])

                report_data.append({
                    'split': split,
                    'file': item['file'],
                    'missing_detections': len(analysis['missing_detections']),
                    'false_positives': len(analysis['false_positives']),
                    'wrong_classifications': len(analysis['wrong_classifications']),
                    'total_gt': len(item['ground_truth']),
                    'total_pred': len(item['predictions'])
                })

        df = pd.DataFrame(report_data)
        df.to_csv(output_path / 'misdetection_summary.csv', index=False)

        with open(output_path / 'misdetection_details.json', 'w') as f:
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_types(misdetections), f, indent=2)
        
        print(f"Misdetection analysis saved to {output_path}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset for duplicates and misdetections')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, 
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--iou_threshold', type=float, default=0.9,
                       help='IoU threshold for duplicate detection (default: 0.9)')
    parser.add_argument('--confidence_threshold', type=float, default=0.25,
                       help='Confidence threshold for model inference (default: 0.25)')
    parser.add_argument('--skip_duplicates', action='store_true',
                       help='Skip duplicate detection analysis')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip model validation analysis')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization of misdetections')
    parser.add_argument('--sample_percentage', type=float, default=100.0,
                       help='Percentage of misdetected images to visualize (0-100, default: 100)')
    
    args = parser.parse_args()
    
    # Validate sample_percentage
    if not 0 <= args.sample_percentage <= 100:
        parser.error("sample_percentage must be between 0 and 100")
    
    analyzer = DatasetAnalyzer(args.dataset_path, args.model_path)
    
    print("Starting dataset analysis...")
    
    # Task 1: Find duplicate detections
    if not args.skip_duplicates:
        print("\n=== TASK 1: Finding Duplicate Detections ===")
        duplicates = analyzer.find_duplicate_detections(args.iou_threshold)
        
        if duplicates:
            duplicate_df = analyzer.save_duplicate_report(duplicates)
            print(f"\nDuplicate Detection Summary:")
            print(duplicate_df.groupby(['split', 'class1', 'class2']).size().to_string())
        else:
            print("No duplicate detections found!")
    
    # Task 2: Model validation with visualization
    if not args.skip_validation and args.model_path:
        print(f"\n=== TASK 2: Model Validation with Visual Analysis ({args.sample_percentage}% of samples) ===")
        misdetections = analyzer.run_model_validation(
            args.confidence_threshold, 
            save_misdetections=True,
            visualize=not args.no_visualize,
            sample_percentage=args.sample_percentage
        )
        
        if misdetections:
            print(f"Model validation completed. Check 'misdetection_analysis' folder for results, visualizations, and annotated images.")
        else:
            print("No misdetections found! Your model performs perfectly on the dataset.")
    elif not args.model_path and not args.skip_validation:
        print("Skipping model validation - no model path provided")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

# Save ALL misdetected images with annotations (100%)
# python dataset_analyzer.py --dataset_path E:/Projects/Python/ComicGlot/training/detection/datasets/cg-manhua-bubbles-10k-v8.yolov12 --model_path E:/Projects/Python/ComicGlot/runs/detect/train18/weights/best.pt --sample_percentage 100

# Save 50% of misdetected images
# python dataset_analyzer.py --dataset_path E:/Projects/Python/ComicGlot/training/detection/datasets/cg-manhua-bubbles-10k-v8.yolov12 --model_path E:/Projects/Python/ComicGlot/runs/detect/train18/weights/best.pt --sample_percentage 50

# Save 25% sample
# python dataset_analyzer.py --dataset_path E:/Projects/Python/ComicGlot/training/detection/datasets/cg-manhua-bubbles-10k-v8.yolov12 --model_path E:/Projects/Python/ComicGlot/runs/detect/train18/weights/best.pt --sample_percentage 25