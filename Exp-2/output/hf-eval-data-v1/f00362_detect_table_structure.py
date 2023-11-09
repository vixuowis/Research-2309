from transformers import pipeline


def detect_table_structure(table_image):
    """
    This function uses the Hugging Face Transformers library to detect the structure of a table in a given image.
    It uses the 'microsoft/table-transformer-structure-recognition' model which is trained on the PubTables1M dataset.
    
    Args:
    table_image (str): Path to the table image file.
    
    Returns:
    dict: The detected table structure.
    """
    # Create an object detection model using the pipeline function
    table_detector = pipeline('object-detection', model='microsoft/table-transformer-structure-recognition')
    
    # Use the model to detect the table structure in the given image
    table_structure = table_detector(table_image)
    
    return table_structure