from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

def get_image_answer(url, question):
    '''
    This function takes an image URL and a question as input, and returns the answer to the question based on the image.
    It uses the Vision-and-Language Transformer (ViLT) model fine-tuned on VQAv2 from Hugging Face Transformers.
    '''
    # Load the image from the given URL
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Load the ViLT model and processor pretrained on VQAv2
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    
    # Tokenize the image and text and create PyTorch tensors
    encoding = processor(image, question, return_tensors='pt')
    
    # Retrieve the output logits from the model
    outputs = model(**encoding)
    logits = outputs.logits
    
    # Find the index with the highest value in logits and convert it to a human-readable answer
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    
    return answer