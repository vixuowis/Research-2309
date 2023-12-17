# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3Tokenizer

# function_code --------------------

def extract_information(image_path, questions):
    """
    Extract information from the document image based on the given questions using LayoutLMv3 model.

    :param image_path: str - the path to the document image file
    :param questions: list - a list of questions to be answered based on the document
    :return: dict - a dictionary containing questions and their corresponding answers
    """
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    answers = {}
    for question in questions:
        input_data = tokenizer(question, image_path, return_tensors='pt')
        output = model(**input_data)
        answer_tokens = tokenizer.convert_ids_to_tokens(
            output.start_logits.argmax(), output.end_logits.argmax() + 1
        )
        answers[question] = ' '.join(answer_tokens)

    return answers

# test_function_code --------------------

def test_extract_information():
    print('Testing started.')
    image_path = 'path/to/test/image.png'
    questions = ['What is the total amount?', 'When is the due date?']

    # Expecting dictionary with answers for provided questions
    expected_keys = set(questions)
    answers = extract_information(image_path, questions)

    assert isinstance(answers, dict), 'The result must be a dictionary.'
    assert set(answers.keys()) == expected_keys, 'The dictionary keys must match the questions.'

    # User would validate if answers make sense
    print('Results:', answers)

    print('Testing finished.')

test_extract_information()