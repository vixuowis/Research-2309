from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from PIL import Image

# Function to detect objects in an image using Deformable DETR model
# @param image_path: Path to the image file
# @return: Object detection results

def detect_objects(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Initialize the image processor and the model
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    
    # Process the image and pass it to the model
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    return outputs