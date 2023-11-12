# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_answers(questions: list, document: str) -> list:
    """
    Extracts answers from a document for a given set of questions using a pretrained model.

    Args:
        questions (list): A list of questions to be answered based on the document.
        document (str): The document from which answers are to be extracted.

    Returns:
        list: A list of answers extracted from the document for each question.

    Raises:
        ImportError: If the required libraries are not found in the environment.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    answers = []
    for question in questions:
        inputs = tokenizer(question, document, return_tensors='pt')
        outputs = model(**inputs)
        start_position = outputs.start_logits.argmax().item()
        end_position = outputs.end_logits.argmax().item()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position+1]))
        answers.append(answer)
    return answers

# test_function_code --------------------

def test_extract_answers():
    """
    Tests the function extract_answers.
    """
    questions = ['What is the capital of France?', 'What currency does France use?']
    document = 'The capital of France is Paris. The country is located in Europe and uses the Euro as its currency.'
    answers = extract_answers(questions, document)
    assert answers == ['Paris', 'Euro'], f'Error: {answers}'
    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_answers()