# function_import --------------------

from transformers import pipeline

# function_code --------------------

def assess_paraphrase_adequacy(paraphrase: str) -> dict:
    """
    Assess the adequacy of a paraphrased text using a pretrained model.

    Args:
        paraphrase (str): The paraphrased text to be assessed.

    Returns:
        dict: The classification result from the model.
    """
    adequacy_classifier = pipeline('text-classification', model='prithivida/parrot_adequacy_model')
    paraphrase_adequacy = adequacy_classifier(paraphrase)
    return paraphrase_adequacy

# test_function_code --------------------

def test_assess_paraphrase_adequacy():
    """
    Test the function assess_paraphrase_adequacy.
    """
    paraphrase1 = 'How can I help you today?'
    paraphrase2 = 'What can I do for you today?'
    paraphrase3 = 'How may I assist you today?'
    assert isinstance(assess_paraphrase_adequacy(paraphrase1), dict)
    assert isinstance(assess_paraphrase_adequacy(paraphrase2), dict)
    assert isinstance(assess_paraphrase_adequacy(paraphrase3), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_assess_paraphrase_adequacy()