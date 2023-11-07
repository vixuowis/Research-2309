from f00440_preprocess_function import *
def test_preprocess_function():
    example = {
        'text': 'This is an example text.'
    }

    # Call the preprocess_function
    preprocessed_example = preprocess_function(example)

    # Check if the text is preprocessed
    assert preprocessed_example['text'] == 'This is a preprocessed example text.'

    print('Test passed!')

test_preprocess_function()
