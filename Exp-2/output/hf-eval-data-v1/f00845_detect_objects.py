def detect_objects(url: str, texts: List[str], model_name: str = 'google/owlvit-large-patch14', score_threshold: float = 0.1):
    """
    Detect objects in an image using the OwlViT model.

    Args:
        url (str): URL of the image.
        texts (List[str]): List of text descriptions.
        model_name (str, optional): Name of the pretrained model. Defaults to 'google/owlvit-large-patch14'.
        score_threshold (float, optional): Threshold for object detection score. Defaults to 0.1.

    Returns:
        List[Dict[str, Union[str, float, List[float]]]]: List of detected objects with their descriptions, confidence scores, and bounding box locations.
    """
    import requests
    from PIL import Image
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    # Load the processor and model
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    # Load the image
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess the image and text descriptions
    inputs = processor(text=texts, images=image, return_tensors='pt')

    # Make predictions
    outputs = model(**inputs)

    # Post-process the outputs
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    # Extract the bounding boxes, scores, and labels
    detections = []
    for i, result in enumerate(results):
        boxes, scores, labels = result['boxes'], result['scores'], result['labels']
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                detections.append({'description': texts[label], 'confidence': round(score.item(), 3), 'location': box})

    return detections