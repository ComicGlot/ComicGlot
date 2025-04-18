from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('yolo11m.pt')  # Load a model
    #model = YOLO('yolov9m.pt')  # Load a model
    model = YOLO('yolo12s.pt')  # Load a model
    #model = YOLO('E:/Projects/Python/Manhua_Scanner/runs/detect/tune/weights/best.pt')
    
    results = model.train(
    data="E:/Projects/Python/Manhua_Scanner/datasets/Manhua_bubbles.v7.yolov11/data.yaml",

        epochs=80,
        #patience=30,  # Stop if no improvement in 10 epochs
        batch=16,  # Batch size
        #imgsz=800,  # Input image size (multiple of max stride 32)
        #lr0=0.01,  # Initial learning rate
        #lrf=0.01,  # Final learning rate fraction
        #weight_decay=0.001,  # Regularization
        augment=True,  # Enable data augmentation

        # lr0=0.0033,
        # lrf=0.0074,
        # momentum=0.93356,
        # weight_decay=0.00064,
        # warmup_epochs=5.0,
        # warmup_momentum=0.94579,
        # box=5.92061,
        # cls=0.62926,
        # dfl=0.88804,
        # hsv_h=0.01001,
        # hsv_s=0.73478,
        # hsv_v=0.33277,
        # translate=0.09296,
        # scale=0.3155,
        # fliplr=0.61522,
        # mosaic=0.55066,
    )