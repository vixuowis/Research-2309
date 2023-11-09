from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image
import pytesseract
from io import BytesIO

def get_document_answer(image_url: str, question: str) -> str:
    """
    This function takes an image URL and a question as input, and returns the answer to the question based on the text in the image.
    It uses the Hugging Face Transformers library and the LayoutLMv2 model for document question-answering tasks.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Extract the text from the image
    text = pytesseract.image_to_string(img)
    
    # Tokenize the text and the question
    inputs = tokenizer(text, question, return_tensors="pt")
    
    # Run the model
    output = model(**inputs)
    
    # Decode the answer
    answer = tokenizer.decode(output["answer_start"][0], output["answer_end"][0])
    
    return answer