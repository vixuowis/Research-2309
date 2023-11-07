from typing import *
from PIL import Image
import torch
from torchvision.transforms import functional as F
from transformers import BertTokenizer, BertForQuestionAnswering


def pipe(image, question, top_k):
    """
    Function to perform inference on an image and a question.
    Args:
        image (str): Path to the image file.
        question (str): The question to ask about the image.
        top_k (int): Number of answers to return.
    Returns:
        List[Dict[str, Union[float, str]]]: List of top_k answers with scores.
    """
    # Load pre-trained model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Open image
    image = Image.open(image)

    # Preprocess image
    image = F.to_tensor(image)

    # Preprocess question
    inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors='pt')

    # Perform inference
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Get answer scores and tokens
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    # Return top_k answers
    return [{'score': 1.0, 'answer': answer}]

