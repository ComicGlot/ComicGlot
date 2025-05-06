from ultralytics import YOLO

# tune5 - dataset v3, model n, 30 epochs, 140/500 iterations
# tune6 - dataset v3, model n, 80 epochs, 300/500 iterations
# tune - dataset v3, model m, 10 epochs, 500/500 iterations

if __name__ == '__main__':
    model = YOLO('yolo11m.pt')  # Load a model

    #model = YOLO('./runs/detect/tune5/weights/last.pt')  # Load a model

    # Define search space
    search_space = {
        # Data augmentation parameters
        "scale": (0.8, 1.2),                    # Rescale images between 80% and 120% of original size
        "translate": (0.0, 0.1),                # Allow small translations (0% to 10%)
        #"exposure": (0.8, 1.2),                 # Adjust exposure between 80% and 120%
        #"noise": (0.0, 0.05),                   # Add up to 5% random noise
        "hsv_h": (0.0, 0.015),                  # Small hue adjustments
        "hsv_s": (0.5, 1.5),                    # Vary saturation from half to 1.5x original
        "hsv_v": (0.5, 1.5),                    # Vary brightness from half to 1.5x original

        # Training parameters
        "lr0": (1e-5, 1e-1),                    # Learning rate range
        "momentum": (0.6, 0.98),                # Momentum for optimizers like SGD
        "weight_decay": (1e-4, 1e-2),           # Regularization range
        "batch_size": (8, 32),                  # Batch size between 8 and 32
        "optimizer": ["Adam", "AdamW", "SGD"],  # Optimizer choices
        "warmup_epochs": (0, 5),                # Gradual learning rate increase over 0 to 5 epochs
        "gradient_clip": (0.0, 1.0),            # Cap gradient values between 0 and 1
    }

    results = model.tune(
        #resume=True,
        data="./datasets/Manhua_bubbles_v3/data.yaml",
        epochs=10,
        iterations=500,
        optimizer="AdamW",
        batch=40,
        #space=search_space,
        #plots=True,
        #save=True,
        #val=True,
        plots=False,
        save=False,
        val=False,
    )

    print("Best hyperparameters:", results['best'])
    print("Best validation mAP:", results['metrics']['map'])