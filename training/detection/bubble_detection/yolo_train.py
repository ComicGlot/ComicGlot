from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    #model = YOLO('yolo11m.pt')  # Load a model
    model = YOLO('yolov9m.pt')  # Load a model
    #model = YOLO('yolo12l.pt')  # Load a model
    #model = YOLO('./runs/detect/tune/weights/best.pt')
    
    results = model.train(
        data="./datasets/cg-manhua-bubbles-10k-v9.yolov12/data.yaml",
        device='cuda',  # Use GPU instead of CPU

        # epochs=80,  # TEST COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #patience=30, 
        batch=48,
        workers=10,

        # ------ Test ------
        epochs=150,  # Up from 80
        close_mosaic=25,
        #cls=2.5,
        copy_paste=0.4,  # Up from 0.3 - helps minority classes
        mixup=0.25,  # Up from 0.15
        # ------------------

        #imgsz=800,  # Input image size (multiple of max stride 32)
        #lr0=0.01,  # Initial learning rate
        #lrf=0.01,  # Final learning rate fraction
        #weight_decay=0.001,  # Regularization

        amp=True,       # Mixed precision training
        augment=True,  # Enable data augmentation

        lr0=0.0033,
        lrf=0.0074,
        momentum=0.93356,
        weight_decay=0.00064,
        warmup_epochs=5.0,
        warmup_momentum=0.94579,
        box=5.92061,
        cls=0.62926, # TEST COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dfl=0.88804,
        hsv_h=0.01001,
        hsv_s=0.73478,
        hsv_v=0.33277,
        translate=0.09296,
        scale=0.3155,
        fliplr=0.61522,
        mosaic=0.55066,

        plots=True,
        save=True,
        verbose=True,
        save_period=10,
        # close_mosaic=10, # TEST COMMENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )