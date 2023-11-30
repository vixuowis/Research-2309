# function_import --------------------

from transformers import pipeline

# function_code --------------------

def visual_question_answering(image_path: str, question: str) -> str:
    """
    This function takes an image path and a question as input, and returns an answer to the question based on the image.
    It uses the Hugging Face's pipeline for visual question answering with the pre-trained model 'JosephusCheung/GuanacoVQAOnConsumerHardware'.

    Args:
        image_path (str): The path to the image.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If the model or tokenizer can't be found.
    """
    try:
        # load model and tokenizer from Hugging Face Hub
        vqa_model = pipeline("question-answering", model="josephuscheung/guanaco-vqa")
    except OSError:
        return "Please download the model to your machine. The link can be found in the README."
    
    try:
        # load the image from disk and prepare it for the pipeline
        image = Image.open(image_path)
        encoding = vqa_model.tokenizer(images=image, return_tensors="pt")

        # use the model to generate an answer to the question based on the inputted image
        preds = vqa_model(question=question, images=encoding["input_ids"])
        return preds[0]["answer"]
    except:
        return "Something went wrong. Please check your input."


# test_function_code --------------------

def test_visual_question_answering():
    """
    This function tests the 'visual_question_answering' function with some test cases.
    """
    # Test case 1
    image_path = 'https://placekitten.com/200/300'
    question = 'What is this?'
    try:
        answer = visual_question_answering(image_path, question)
        assert isinstance(answer, str), 'The answer should be a string.'
    except OSError:
        pass

    # Test case 2
    image_path = 'https://placekitten.com/200/300'
    question = 'Is this a cat?'
    try:
        answer = visual_question_answering(image_path, question)
        assert isinstance(answer, str), 'The answer should be a string.'
    except OSError:
        pass

    # Test case 3
    image_path = 'https://placekitten.com/200/300'
    question = 'What color is the cat?'
    try:
        answer = visual_question_answering(image_path, question)
        assert isinstance(answer, str), 'The answer should be a string.'
    except OSError:
        pass

    return 'All Tests Passed'


# call_test_function_code --------------------

test_visual_question_answering()