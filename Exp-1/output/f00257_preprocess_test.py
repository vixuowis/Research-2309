from f00257_preprocess import *
def test_preprocess():
    question = 'What is the capital of France?'
    context = 'Paris is the capital of France.'

    preprocessed_data = preprocess(question, context)

    assert isinstance(preprocessed_data, dict)
    assert 'input_ids' in preprocessed_data
    assert 'attention_mask' in preprocessed_data

    print('All tests passed!')

test_preprocess()
