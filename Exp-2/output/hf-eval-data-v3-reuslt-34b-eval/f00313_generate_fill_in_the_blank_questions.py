# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_fill_in_the_blank_questions(masked_sentence):
    """
    Generate fill-in-the-blank questions by predicting the masked token in a sentence.

    Args:
        masked_sentence (str): The sentence with a keyword replaced by the '[MASK]' token.

    Returns:
        list: A list of dictionaries. Each dictionary contains a 'score', 'sequence', 'token', and 'token_str' which represent the confidence score, the complete sentence, the token id, and the token string respectively.
    """    
    # Load the model for fill-in-the-blank predictions.
    unmasker = pipeline("fill-mask", model="bert-base-uncased")

    # Make a prediction based on the masked sentence.
    predictions = unmasker(masked_sentence)
    
    # Store the output in a list of dictionaries and return it
    output_list = []
    for prediction in predictions:
        output = {'score': prediction['score'], 'sequence': prediction['sequence'], 'token': prediction['token'], 'token_str': prediction['token_str']}
        output_list.append(output)
    
    return output_list


# test_function_code --------------------

def test_generate_fill_in_the_blank_questions():
    """
    Test the function generate_fill_in_the_blank_questions.
    """
    test_sentence_1 = 'Hello, I am a [MASK] model.'
    test_sentence_2 = 'I love to [MASK] books.'
    test_sentence_3 = 'The [MASK] is shining brightly.'

    result_1 = generate_fill_in_the_blank_questions(test_sentence_1)
    result_2 = generate_fill_in_the_blank_questions(test_sentence_2)
    result_3 = generate_fill_in_the_blank_questions(test_sentence_3)

    assert isinstance(result_1, list) and isinstance(result_1[0], dict)
    assert isinstance(result_2, list) and isinstance(result_2[0], dict)
    assert isinstance(result_3, list) and isinstance(result_3[0], dict)

    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_fill_in_the_blank_questions()