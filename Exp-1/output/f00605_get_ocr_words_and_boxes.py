from typing import *
import cv2
import pytesseract

def get_ocr_words_and_boxes(example):
    # Load the image from the example
    image_path = example['image_path']
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to extract the bounding boxes and text for each word
    boxes = pytesseract.image_to_boxes(gray)
    words = []
    # Parse the boxes to extract the words
    for box in boxes.splitlines():
        _, word, _, _, _ = box.split()
        words.append(word)
    # Add the extracted words and boxes to the example
    example['ocr_words'] = words
    example['ocr_boxes'] = boxes
    return example
