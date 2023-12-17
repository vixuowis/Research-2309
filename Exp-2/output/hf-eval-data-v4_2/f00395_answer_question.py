# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def answer_question(question, context):
    """
    Answer a user's question based on the provided context using a pre-trained NLP model.

    Args:
        question (str): The question posed by the user.
        context (str): The context within which the question should be answered.

    Returns:
        str: The answer to the question extracted from the context, if found.

    Raises:
        ValueError: If `question` or `context` is empty.

    """
    if not question or not context:
        raise ValueError('The `question` and `context` cannot be empty')

    model = AutoModelForQuestionAnswering.from_pretrained('deepset/deberta-v3-large-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')

    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    output = model(**inputs)

    answer_start = output.start_logits.argmax().item()
    answer_end = output.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1])).strip()

    return answer

# test_function_code --------------------

def test_answer_question():
    print('Testing started.')
    # Mock questions and contexts for testing
    mock_questions = [
        'What is the purpose of life?',
        'What is the capital of France?',
        'How many legs does a spider have?'
    ]
    mock_contexts = [
        'Many philosophers and scientists consider the purpose of life as a philosophical question concerning the significance of existence.',
        'France is a country whose territory consists of metropolitan France in Western Europe, as well as several overseas regions and territories. The capital of France is Paris.',
        'Spiders are air-breathing arthropods that have eight legs and chelicerae with fangs able to inject venom.'
    ]
    mock_answers = [
        'philosophical question concerning the significance of existence',
        'Paris',
        'eight'
    ]

    for i, (question, context, expected_answer) in enumerate(zip(mock_questions, mock_contexts, mock_answers), 1):
        print(f'Testing case [{i}/3] started.')
        answer = answer_question(question, context)
        assert answer == expected_answer, f'Test case [{i}/3] failed: expected {{expected_answer}}, got {{answer}}'

    print('Testing finished.')

# call_test_function_line --------------------

test_answer_question()