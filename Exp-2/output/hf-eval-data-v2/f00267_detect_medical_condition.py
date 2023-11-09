# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def detect_medical_condition(image, question):
    """
    This function uses a pretrained model from Hugging Face Transformers to detect the medical condition present in an image.
    
    Args:
        image (PIL.Image): The image to be analyzed.
        question (str): The question to be asked to the model about the image. It should specifically ask about the medical condition in the image.
    
    Returns:
        str: The detected medical condition.
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
    This function tests the 'detect_medical_condition' function with a sample image and question.
    """
    sample_image = 'sample_image.jpg'  # Replace with a valid image file path
    sample_question = 'What medical condition is present in the image?'
    detected_condition = detect_medical_condition(sample_image, sample_question)
    assert isinstance(detected_condition, str), 'The function should return a string.'

# call_test_function_code --------------------

test_detect_medical_condition()