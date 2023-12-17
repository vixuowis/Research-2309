# requirements_file --------------------

!pip install -U transformers==4.12.2 torch==1.8.0+cu101 datasets==1.14.0 tokenizers==0.10.3

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_answers_from_document(questions, document):
    """
    Extract answers for a list of questions from a given document using a fine-tuned document question answering model.

    Args:
        questions (list of str): A list of questions to be answered.
        document (str): The document containing the information to answer the questions.

    Returns:
        dict: A dictionary mapping each question to its corresponding answer.

    Raises:
        ValueError: If no questions are provided or if the document is empty.
    """

    if not questions or not document:
        raise ValueError("Questions or document cannot be empty.")

    # Load the pre-trained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    # Prepare to collect answers
    answers = {}

    # Process each question
    for question in questions:
        inputs = tokenizer(question, document, return_tensors="pt")
        outputs = model(**inputs)
        start_position = outputs.start_logits.argmax().item()
        end_position = outputs.end_logits.argmax().item()
        answer = tokenizer.decode(inputs['input_ids'][0][start_position:end_position + 1], skip_special_tokens=True)
        answers[question] = answer

    return answers


# test_function_code --------------------

def test_extract_answers_from_document():
    print("Testing started.")
    questions = ["What is the capital of France?"]
    document = "The capital of France is Paris. The country is located in Europe and uses the Euro as its currency."

    # Testing case 1: Valid questions and document
    print("Testing case [1/1] started.")
    answers = extract_answers_from_document(questions, document)
    assert answers['What is the capital of France?'] == 'Paris', f"Test case [1/1] failed: Expected 'Paris', got {answers['What is the capital of France?']}"
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_answers_from_document()