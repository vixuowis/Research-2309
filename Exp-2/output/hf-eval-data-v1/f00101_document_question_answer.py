from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import cv2
import torch

def document_question_answer(image_path: str, question: str) -> str:
    '''
    This function takes an image path and a question as input, and returns the answer to the question based on the content of the image.
    The image is supposed to be a scanned document.
    The function uses a pre-trained model from Hugging Face Transformers to perform document question answering.
    '''
    # Load the pre-trained model and tokenizer
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)

    # Read and preprocess the image
    image = cv2.imread(image_path)

    # Use the tokenizer to create input tokens
    input_tokens = tokenizer(question, image, return_tensors='pt')

    # Feed the input tokens into the model
    output = model(**input_tokens)

    # Extract the predicted answer from the model's output
    start_logits, end_logits = output.start_logits, output.end_logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)
    answer = tokenizer.decode(input_tokens['input_ids'][0][answer_start:answer_end + 1])

    return answer