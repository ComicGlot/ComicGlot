from ultralytics import YOLO
import os

#train6  - test_set 1, train_set v2, model yolo11m, 100 epochs - predict1
#train8  - test_set 1, train_set v2, model yolo11x, 200 epochs - predict2
#train9  - test_set 1, train_set v2, model yolo11x, 20 epochs  - predict3
#train11 - test_set 1, train_set v2, model yolo11x, 80 epochs  - predict4
#train11 - test_set 2, train_set v2, model yolo11x, 80 epochs  - predict5
#train11 - test_set 2, train_set v2, model yolo11x, 80 epochs  - predict6
#train12 - test_set 1, train_set v3, model yolo11x, 80 epochs  - predict7
#train12 - test_set 2, train_set v3, model yolo11x, 80 epochs  - predict8
#train13 - test_set 1, train_set v3, model yolo11n, 80 epochs  - predict9
#train13 - test_set 2, train_set v3, model yolo11n, 80 epochs  - predict10
#train13 - test_set 3, train_set v3, model yolo11n, 80 epochs  - predict11
#train14 - test_set 1, train_set v5, model yolo11m, 80 epochs  - predict12
#train14 - test_set 2, train_set v5, model yolo11m, 80 epochs  - predict13
#train14 - test_set 3, train_set v5, model yolo11m, 80 epochs  - predict14
#train14 - test_set 4, train_set v5, model yolo11m, 80 epochs  - predict15
#train15 - test_set 4, train_set v5, model yolo11m, 200 epochs - predict16
#train16 - test_set 3, train_set v6, model yolo11m, 80 epochs  - predict17
#train17 - test_set 3, train_set v6, model yolo11m, 120 epochs - predict18
#train18 - test_set 5, train_set v7, model yolo11m, 80 epochs  - predict19
#train18 - test_set 4, train_set v7, model yolo11m, 80 epochs  - predict20
#train19 - test_set 3,  train_set v7, model yolo11m, 80 epochs  - predict21 finetuned
#train20 - test_set ,  train_set v7, model yolo11m, 120 epochs - predict finetuned
#train21 - test_set ,  train_set v7, model yolo11m, 70 + 10 epochs - predict finetuned on best.pt

#train22 - test_set , train_set v7, model yolo9m, 80 epochs  - predict best tuned
#train23 - test_set , train_set v7, model yolo12m, 80 epochs  - predict best tuned
#train24 - test_set , train_set v7, model yolo12m, 80 epochs  - predict
#train25 - test_set , train_set v7, model yolo12s, 80 epochs  - predict

# Needed:

# YOLO12 (tuned)
# YOLO11 (tuned)     V
# YOLO11
# YOLO9 (tuned)      V
# DETR ResNet 50
# 

if __name__ == '__main__':
    model = YOLO("E:/Projects/Python/Manhua_Scanner/runs/detect/train19/weights/best.pt")
    
    folder = "E:/Projects/Python/Manhua_Scanner/datasets/testing/3/"
    output_folder = "E:/Projects/Python/Manhua_Scanner/datasets/predictions/1"

    #os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(folder):
        relative_path = os.path.relpath(root, folder)
        current_output_folder = os.path.join(output_folder, relative_path)
        os.makedirs(current_output_folder, exist_ok=True)

        for image_name in files:
            image_path = os.path.join(root, image_name)

            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing {image_path}")

            # results = model(image_path, save=True, save_crop=True, iou=0.5, conf=0.75)
            results = model(image_path, save=True, save_crop=True)