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
    
    # Load the pretrained pipeline model
    fill_mask = pipeline(task='fill-mask')
    results = [] 

    if masked_sentence == "":
        return "No sentences provided!"
    else:
        for i in range(5):
            result = {"score" : 0, "sequence" : "", "token" : "", "token_str" : ""} 
            prediction = fill_mask(masked_sentence)
            
            # Extract the top prediction's score and token id from the model's output
            result['score'] = float('{:.3f}'.format(prediction[0]['score']))
            result['token'] = int(prediction[0]['token'])
            result['sequence'] = masked_sentence
            
            # Use token id to get the string from the model's tokenizer
            result['token_str'] = str(fill_mask.tokenizer.decode([result["token"]])).replace('<pad>', '').replace('</s>','')
            results.append(result) 
            
        return results
# function_code --------------------

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