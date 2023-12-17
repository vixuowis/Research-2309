# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# function_code --------------------

def answer_visual_questions(question_text, image_path_or_url):
    """
    Answer questions about an image using a pretrained AI model.

    Args:
        question_text (str): The question text.
        image_path_or_url (str): The file path or URL to the image.

    Returns:
        str: The predicted answer to the question based on the image.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/git-large-textvqa")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/git-large-textvqa")

    image_question_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer
    )
    answer = image_question_pipeline(question=question_text, image=image_path_or_url)
    return answer

# test_function_code --------------------

def test_answer_visual_questions():
    print("Testing started.")
    # Test case with a hypothetical image URL and question
    image_url = "https://example.com/sample_image.jpg"
    question = "What is written on the sign in the image?"

    print("Testing single case...")
    answer = answer_visual_questions(question, image_url)
    assert isinstance(answer, str), f"Expected a string answer, but got: {type(answer)}"
    print("Test passed.")

# Run the test function
test_answer_visual_questions()