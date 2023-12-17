# requirements_file --------------------

!pip install -U transformers==4.12.2 pytorch==1.8.0+cu101 datasets==1.14.0 tokenizers==0.10.3

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def document_question_answering(question, document):
    """
    Answer a question based on a given document using a pre-trained model.

    Parameters:
        question (str): The question to be answered.
        document (str): The document containing the answer.

    Returns:
        str: The predicted answer from the document.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    input_dict = tokenizer(question, document, return_tensors='pt')
    output = model(**input_dict)
    answer = tokenizer.convert_ids_to_tokens(output['answer_ids'][0], skip_special_tokens=True)
    return ' '.join(answer)

# test_function_code --------------------

def test_document_question_answering():
    print("Testing started.")
    # Example document and question
    document = "..."
    question = "..."

    # Expected answer prefix to test if the function works
    expected_answer_prefix = "..."
    answer = document_question_answering(question, document)
    assert answer.startswith(expected_answer_prefix), f"Test failed: Expected answer to start with '{{expected_answer_prefix}}' but got '{{answer}}'"
    print("Testing case [1/1] succeeded.")
    print("Testing finished.")