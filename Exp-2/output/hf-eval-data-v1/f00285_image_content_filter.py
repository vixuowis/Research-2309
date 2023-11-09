from transformers import pipeline


def image_content_filter(image_path):
    """
    This function uses a zero-shot classification model to classify images into predefined categories
    like 'safe for work', 'adult content', or 'offensive'. It then filters out images that are classified
    as adult content or offensive.

    Parameters:
    image_path (str): The path to the image or an image URL

    Returns:
    str: The classification result
    """
    # Create a zero-shot classification model
    image_classifier = pipeline('zero-shot-classification', model='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    class_names = ['safe for work', 'adult content', 'offensive']
    # Classify the image
    result = image_classifier(image=image_path, class_names=class_names)
    # Filter out images that are classified as adult content or offensive
    if result['labels'][0] in ['adult content', 'offensive']:
        return 'Filtered'
    else:
        return 'Passed'