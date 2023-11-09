# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def document_question_answer(document_text):
    """
    This function implements an optical text recognition system for documents that can answer a simple question about the document's content.
    It uses the pretrained model 'DataIntelligenceTeam/eurocorpV4' from Hugging Face Transformers.

    Args:
        document_text (str): The text of the document to be analyzed.

    Returns:
        token_classification_results (numpy.ndarray): The classified tokens from the document text.
    """
    model = AutoModelForTokenClassification.from_pretrained('DataIntelligenceTeam/eurocorpV4')
    tokenizer = AutoTokenizer.from_pretrained('DataIntelligenceTeam/eurocorpV4')
    inputs = tokenizer(document_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    token_classification_results = outputs.logits.argmax(-1).numpy()
    return token_classification_results

# test_function_code --------------------

def test_document_question_answer():
    """
    This function tests the document_question_answer function.
    It uses a sample document text and checks if the output is a numpy.ndarray.
    """
    sample_document_text = 'This is a sample document text.'
    result = document_question_answer(sample_document_text)
    assert isinstance(result, np.ndarray), 'The result should be a numpy.ndarray.'

# call_test_function_code --------------------

test_document_question_answer()