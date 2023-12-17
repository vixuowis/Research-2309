# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def load_korean_qa_pipeline():
    """
    Load the Korean language question-answering model.

    Returns:
        A callable pipeline object that can be used to perform question-answering
        on Korean texts.
    """
    korean_qa_pipeline = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    return korean_qa_pipeline

# test_function_code --------------------

def test_load_korean_qa_pipeline():
    print('Testing load_korean_qa_pipeline function.')
    qa_pipeline = load_korean_qa_pipeline()
    assert callable(qa_pipeline), 'The function should return a callable pipeline.'
    print('Test for load_korean_qa_pipeline passed.')

test_load_korean_qa_pipeline()