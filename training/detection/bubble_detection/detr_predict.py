from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import os
from PIL import Image
import requests
from dotenv import load_dotenv
import cv2
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

MODEL_PATH = os.path.abspath('./detr-resnet-50-bubbles-10ep')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

INPUT_IMAGE = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))
image = cv2.imread(INPUT_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct visualization

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(DEVICE)

inputs = processor(images=image, return_tensors="pt").to(DEVICE)
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.shape[:2]], dtype=torch.float32).to(DEVICE)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
print(results)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

# Plot image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_name = model.config.id2label[label.item()]

    # Draw bounding box
    x_min, y_min, x_max, y_max = box
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, color='red', linewidth=2))

    # Add label text
    text = f"{class_name}: {round(score.item(), 2)}"
    plt.text(x_min, y_min - 5, text, fontsize=12, color='red', backgroundcolor='white')

plt.axis("off")
plt.show()
