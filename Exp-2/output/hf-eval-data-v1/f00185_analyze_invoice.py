from transformers import LayoutLMForQuestionAnswering
from PIL import Image

# Function to analyze customer invoices
# This function uses the Hugging Face Transformers library to load a pretrained model for question answering
# The model is capable of processing images and extracting relevant information
# The function takes as input the path to an image file containing an invoice and a question
# It returns the answer to the question as provided by the model

def analyze_invoice(image_path: str, question: str) -> str:
    # Initialize a question-answering pipeline with the loaded model
    nlp = pipeline('question-answering', model=LayoutLMForQuestionAnswering.from_pretrained('microsoft/layoutlm-base-uncased'))
    
    # Pass the image file and the question as input to the pipeline
    result = nlp(question, image_path)
    
    # Return the answer
    return result['answer']