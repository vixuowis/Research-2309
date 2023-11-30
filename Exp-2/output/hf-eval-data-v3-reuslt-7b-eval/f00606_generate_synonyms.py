# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_synonyms(word):
    """
    Generate synonyms for a given word using the 'microsoft/deberta-base' model.

    Args:
        word (str): The word to generate synonyms for.

    Returns:
        list: A list of synonyms for the given word.
    """
    
    # load a pretrained model from Huggingface --------------------
    synonym_extractor = pipeline("sentence-transformers/paraphrase-MiniLM-L12-v2")
    
    # generate synonyms for the given word --------------------
    return synonym_extractor(word, max_length=30)[0]['paraphrases'][1:]

# test_function_code --------------------

def test_generate_synonyms():
    """
    Test the generate_synonyms function.
    """
    synonyms = generate_synonyms('happy')
    assert isinstance(synonyms, list)
    assert len(synonyms) > 0
    assert all(isinstance(synonym, str) for synonym in synonyms)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_synonyms()