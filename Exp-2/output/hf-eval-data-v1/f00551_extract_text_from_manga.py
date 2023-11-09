from transformers import pipeline
import os


def extract_text_from_manga(image_path):
    """
    This function extracts Japanese text from a manga page image using the Hugging Face Transformers OCR pipeline.
    
    Args:
        image_path (str): The path to the manga page image.
    
    Returns:
        str: The extracted text.
    """
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    # Create an OCR pipeline using the 'kha-white/manga-ocr-base' model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    
    # Use the OCR pipeline to extract text from the manga page image
    extracted_text = ocr_pipeline(image_path)
    
    return extracted_text