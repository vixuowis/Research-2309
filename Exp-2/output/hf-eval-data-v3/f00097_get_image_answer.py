# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# function_code --------------------

def get_image_answer(question_text: str, image_path_or_url: str) -> str:
    """
    This function uses a pretrained model from Hugging Face Transformers to answer questions about an image.

    Args:
        question_text (str): The question about the image.
        image_path_or_url (str): The path or URL of the image.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/git-large-textvqa')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textvqa')

    image_question_pipeline = pipeline(
        'question-answering', model=model, tokenizer=tokenizer
    )
    answer = image_question_pipeline(question=question_text, image=image_path_or_url)
    return answer

# test_function_code --------------------

def test_get_image_answer():
    """
    This function tests the get_image_answer function.
    """
    question_text = 'What color is the cat?'
    image_path_or_url = 'https://placekitten.com/200/300'
    answer = get_image_answer(question_text, image_path_or_url)
    assert answer is not None, 'The function did not return an answer.'

    question_text = 'Is the cat sitting or standing?'
    image_path_or_url = 'https://placekitten.com/200/300'
    answer = get_image_answer(question_text, image_path_or_url)
    assert answer is not None, 'The function did not return an answer.'

    question_text = 'Does the cat have stripes?'
    image_path_or_url = 'https://placekitten.com/200/300'
    answer = get_image_answer(question_text, image_path_or_url)
    assert answer is not None, 'The function did not return an answer.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_image_answer()