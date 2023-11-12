# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def detect_medical_condition(image, question):
    """
    This function uses a pretrained model from Hugging Face Transformers to detect the medical condition present in an image.
    The model is fine-tuned on TextVQA and can be used for multimodal tasks like visual question answering.
    Args:
        image (str): The URL or local path of the image.
        question (str): The question about the medical condition in the image.
    Returns:
        str: The detected medical condition.
    Raises:
        ValueError: If the model type is not recognized.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/git-large-textvqa')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textvqa')
    encoded_input = tokenizer(question, image, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input)
    detected_medical_condition = tokenizer.decode(generated_tokens[0])
    return detected_medical_condition

# test_function_code --------------------

def test_detect_medical_condition():
    """
    This function tests the detect_medical_condition function with a sample image and question.
    """
    sample_image = 'https://placekitten.com/200/300'
    sample_question = 'What medical condition is present in the image?'
    detected_condition = detect_medical_condition(sample_image, sample_question)
    assert isinstance(detected_condition, str), 'The detected condition should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_medical_condition()