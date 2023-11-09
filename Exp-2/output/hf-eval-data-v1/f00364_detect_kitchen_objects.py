def detect_kitchen_objects(image_path, score_threshold=0.1):
    '''
    This function detects different objects in a kitchen using a pre-trained model from Hugging Face Transformers.
    It uses the OwlViTForObjectDetection model which is trained for zero-shot text-conditioned object detection tasks.
    The function takes an image path and a score threshold as input, and returns the detected objects with their confidence scores and locations.
    '''
    from PIL import Image
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    
    # Load the processor and model
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    
    # Load the image
    image = Image.open(image_path)
    
    # Define the text queries
    texts = [["a photo of a fruit", "a photo of a dish"]]
    
    # Process the image and text queries
    inputs = processor(text=texts, images=image, return_tensors='pt')
    
    # Get the model outputs
    outputs = model(**inputs)
    
    # Post-process the outputs
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    # Initialize an empty list to store the detections
    detections = []
    
    # Loop over the results and add the detections to the list
    for i in range(len(texts)):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                detections.append((texts[0][label], round(score.item(), 3), box))
    
    return detections