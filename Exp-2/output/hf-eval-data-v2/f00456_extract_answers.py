# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_answers(questions, document):
    """
    Extracts answers from a given document for a set of questions using a pretrained model.

    Args:
        questions (list): A list of questions to be answered.
        document (str): The document from which to extract the answers.

    Returns:
        dict: A dictionary where keys are questions and values are corresponding answers.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    answers = {}
    for question in questions:
        inputs = tokenizer(question, document, return_tensors="pt")
        outputs = model(**inputs)
        start_position = outputs.start_logits.argmax().item()
        end_position = outputs.end_logits.argmax().item()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_position:end_position+1]))
        answers[question] = answer
    return answers

# test_function_code --------------------

def test_extract_answers():
    """
    Tests the extract_answers function.
    """
    questions = ["What is the capital of France?"]
    document = "The capital of France is Paris. The country is located in Europe and uses the Euro as its currency."
    answers = extract_answers(questions, document)
    assert answers[questions[0]] == "Paris"

# call_test_function_code --------------------

test_extract_answers()