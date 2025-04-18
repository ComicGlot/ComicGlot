import os
import cv2
import logging
from prediction_model import YOLOPredictor
from ocr_pipeline import OCRProcessor
from dotenv import load_dotenv
from utils import crop_image, load_image, draw_bounding_box_with_text, get_next_run_folder, apply_separator_mask
from natsort import natsorted
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)d',
    #level=logging.DEBUG
)

# Load environment variables from .env file
load_dotenv()

YOLO_MODEL_PATH = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
TROCR_MODEL_PATH = os.path.abspath(os.getenv("TROCR_MODEL_PATH", "./models/TrOCR/tuned"))
INPUT_FOLDER = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/chapters/"))
OUTPUT_FOLDER = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
LANGUAGES_STR = os.getenv("LANGUAGE", "ko,en")
LANGUAGES = LANGUAGES_STR.split(',')

def process_folder(input_folder: str, model_path: str, lang: str, output_folder: str = None,
                   save: bool = False, save_bubble_crop: bool = False, save_line_crop: bool = False):
    """
    Process all images in a folder, run YOLO predictions, and perform OCR.
    
    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder containing images.
        model_path (str): Path to the YOLO model.
        lang (str): Language code for OCR.

    Returns:
        list: A list of results combining YOLO predictions and OCR output.
    """
    logging.info(f"Initializing YOLO model from {model_path}")
    yolo = YOLOPredictor(model_path)

    logging.info("Initializing OCR Processor")
    ocr = OCRProcessor(language=lang, custom_model_path=TROCR_MODEL_PATH) #

    results_list = []
    image_count = 1

    if output_folder is None:
        output_folder = os.path.abspath("./models/YOLO/best.pt")
    run_folder = get_next_run_folder(output_folder)
    logging.info(f"Saving outputs to: {run_folder}")

    if save_bubble_crop:
        bubble_output_dir = os.path.join(run_folder, "bubble_crops")
        os.makedirs(bubble_output_dir, exist_ok=True)
    if save_line_crop:
        line_output_dir = os.path.join(run_folder, "line_crops")
        os.makedirs(line_output_dir, exist_ok=True)
    if save:
        annotated_output_dir = os.path.join(run_folder, "annotated_images")
        os.makedirs(annotated_output_dir, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        # Natural sort for file names
        sorted_files = natsorted(files)

        for image_name in sorted_files:
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            image_path = os.path.join(root, image_name)
            logging.info(f"Processing image: {image_path}")
            original_image = load_image(image_path)

            # Run YOLO predictions
            yolo_results, separator_lines = yolo.predict(image_path)
            print(separator_lines)

            for box_number, ((box, cls_name, confidence), separator_line) in enumerate(zip(yolo_results, separator_lines)):
                x_min, y_min, x_max, y_max = map(int, box)
                cropped_block = crop_image(image_path, (x_min, y_min, x_max, y_max))

                # Apply separator mask if it exists
                if separator_line is not None:
                    cropped_block = apply_separator_mask(cropped_block, separator_line)
                    
                    # Plot the cropped block
                    plt.figure(figsize=(5, 5))
                    plt.imshow(cropped_block, cmap='gray')  # Use 'gray' for better visualization
                    plt.title("Cropped Block with Separator")
                    
                    # Plot the separator line
                    print(separator_line)
                    p1, p2 = separator_line  # Extract points
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)  # Draw red separator line
                    
                    plt.show()

                # Perform OCR on cropped region
                # Pipeline by class 
                if cls_name == "sfx_sound":
                    # Some processing
                    ocr_results = ocr.perform_detection_and_recognition(cropped_block, min_block_size=40, detection=False)
                else: 
                    ocr_results = ocr.perform_detection_and_recognition(cropped_block, min_block_size=40)

                # Combine text for the current cropped block
                block_text = " ".join([text for bbox, text in ocr_results]).strip()

                results_list.append({
                    "image": image_name,
                    "box_number": box_number,
                    "class_name": cls_name,
                    "confidence": round(confidence, 4),
                    "text": block_text
                })

                if save_bubble_crop:
                    # Save cropped_block as img
                    bubble_crop_path = os.path.join(bubble_output_dir, f"{image_name}_bubble_{box_number}.png")
                    cv2.imwrite(bubble_crop_path, cropped_block)
                    # logging.info(f"Saved bubble crop: {bubble_crop_path}")

                if save_line_crop:
                    # Save each box line passed from ocr_results as img
                    for line_number, (bbox, _) in enumerate(ocr_results, start=1):
                        line_x_min, line_y_min, line_x_max, line_y_max = map(int, bbox)
                        cropped_line = crop_image(cropped_block, (line_x_min, line_y_min, line_x_max, line_y_max))
                        line_crop_path = os.path.join(line_output_dir, f"{image_name}_line_{box_number}_{line_number}.png")
                        cv2.imwrite(line_crop_path, cropped_line)
                        # logging.info(f"Saved line crop: {line_crop_path}")

                if save:
                    original_image = draw_bounding_box_with_text(
                        original_image, box, cls_name, confidence, ocr_results
                    )

            if save:
                # Save provided frames with:
                # * Drawn bubble boxes with written confidence and class name
                # * Drawn string boxes with found text on it
                annotated_path = os.path.join(annotated_output_dir, f"annotated_{image_name}")
                cv2.imwrite(annotated_path, original_image)
                logging.info(f"Saved annotated image: {annotated_path}")

            image_count += 1

    # Write down results_list in txt file in output folder
    output_txt_path = os.path.join(run_folder, "results.txt")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for result in results_list:
            f.write(str(result) + "\n")

    return results_list


if __name__ == '__main__':
    logging.info("Starting folder processing pipeline")

    #try:
    results = process_folder(INPUT_FOLDER, 
                             YOLO_MODEL_PATH, 
                             LANGUAGES, 
                             output_folder=OUTPUT_FOLDER,
                             save=True, 
                             save_bubble_crop=True, 
                             save_line_crop=True
                             )

        # Output results
    for result in results:
        logging.info(result)

    # except Exception as e:
    #     logging.error(f"An error occurred: {e}")
