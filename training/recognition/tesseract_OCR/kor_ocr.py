import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update this

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image. Check the file path: {image_path}")

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    scale_percent = 200  # Scale by 200%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return image

def perform_img_ocr(image_path, lang='kor'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    preprocessed_image = preprocess_image(image_path)
    config = '--psm 12 --oem 3'
    extracted_text = pytesseract.image_to_string(preprocessed_image, lang=lang, config=config)
    
    print("Extracted Text:\n", extracted_text)

    parsed_text = extracted_text.strip().replace("\n", "").replace("|", "")
    print(parsed_text)
    cv2.imwrite('preprocessed_image.png', preprocessed_image)  # Save for debugging

def perform_fldr_ocr(folder, lang='kor'):
    for root, _, files in os.walk(folder):
        for image_name in files:
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            image_path = os.path.join(root, image_name)
            print(f"Processing {image_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at: {image_path}")

            preprocessed_image = preprocess_image(image_path)
            #config = '--psm 12 --oem 3'
            config = '--psm 8 --oem 3'

            #   Page segmentation modes (PSM):
            #   0    Orientation and script detection (OSD) only.
            #   1    Automatic page segmentation with OSD.
            #   2    Automatic page segmentation, but no OSD, or OCR.
            #   3    Fully automatic page segmentation, but no OSD. (Default)
            #   4    Assume a single column of text of variable sizes.
            #   5    Assume a single uniform block of vertically aligned text.
            #   6    Assume a single uniform block of text.
            #   7    Treat the image as a single text line.
            #   8    Treat the image as a single word.
            #   9    Treat the image as a single word in a circle.
            #  10    Treat the image as a single character.
            #  11    Sparse text. Find as much text as possible in no particular order.
            #  12    Sparse text with OSD.
            #  13    Raw line. Treat the image as a single text line,
            #                         bypassing hacks that are Tesseract-specific.

            extracted_text = pytesseract.image_to_string(preprocessed_image, lang=lang, config=config)
            
            print("Extracted Text:\n", extracted_text)

            parsed_text = extracted_text.strip().replace("\n", "").replace("|", "")
            print(parsed_text)
            cv2.imwrite('preprocessed_image.png', preprocessed_image)  # Save for debugging

if __name__ == "__main__":
    image_path = r"E:/Projects/Python/Manhua_Scanner/data/bubbles/test2.jpg"
    perform_img_ocr(image_path, lang='kor')
    #perform_fldr_ocr(image_path, lang='kor')
