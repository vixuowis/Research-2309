from transformers import pipeline


def extract_text_from_manga(manga_image):
    """
    This function is used to extract text from a manga image using the 'kha-white/manga-ocr-base' model.
    
    Parameters:
    manga_image (Image): The manga image from which the text is to be extracted.
    
    Returns:
    str: The extracted text from the manga image.
    """
    # Create an OCR pipeline using the 'kha-white/manga-ocr-base' model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    
    # Pass the manga image as input to the pipeline
    manga_text = ocr_pipeline(manga_image)
    
    return manga_text