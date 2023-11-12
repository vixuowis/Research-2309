# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_correct_answer(summary_text: str, question: str, options: list) -> str:
    """
    This function uses a pre-trained BERT model to find the correct answer among multiple options.

    Args:
        summary_text (str): The summary of the article.
        question (str): The question based on the summary.
        options (list): The list of possible answers.

    Returns:
        str: The correct answer among the options.
    """
    # Instantiate the Question Answering pipeline
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')

    # Check the correct answer among the multiple options
    predictions = []
    for option in options:
        result = qa_pipeline({'context': summary_text, 'question': f'{question} {option}'})
        predictions.append((option, result['score']))

    # The highest-scoring option is the correct answer
    correct_answer = max(predictions, key=lambda x: x[1])[0]
    return correct_answer

# test_function_code --------------------

def test_get_correct_answer():
    """
    This function tests the `get_correct_answer` function.
    """
    summary_text = 'The sky is blue because of the way Earth's atmosphere scatters sunlight.'
    question = 'Why is the sky blue?'
    options = ['Because of the ocean', 'Because of sunlight scattering', 'Because of the moon', 'Because of pollution']
    correct_answer = get_correct_answer(summary_text, question, options)
    assert correct_answer == 'Because of sunlight scattering', f'Error: {correct_answer}'

    summary_text = 'Python is an interpreted, high-level and general-purpose programming language.'
    question = 'What is Python?'
    options = ['A snake', 'A programming language', 'A car', 'A movie']
    correct_answer = get_correct_answer(summary_text, question, options)
    assert correct_answer == 'A programming language', f'Error: {correct_answer}'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_correct_answer()