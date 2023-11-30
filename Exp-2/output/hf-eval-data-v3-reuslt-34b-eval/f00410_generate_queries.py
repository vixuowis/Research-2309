# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_queries(document: str) -> str:
    '''
    Generate possible user queries for a given document using a pre-trained T5 model.

    Args:
        document (str): The input document for which to generate queries.

    Returns:
        str: The generated queries.

    Raises:
        ValueError: If the input document is not a string.
    '''
    
    if type(document) != str:
        raise ValueError('Input document must be of type "str"')
        
    elif len(document) == 0:
        return ''
        
    else:

        # load model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_large_paraphraser')
        tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_large_paraphraser')

        # encode input document
        input_ids = tokenizer(document, return_tensors='pt').input_ids
        
        # decode paraphrases
        input_ids = input_ids.to('cuda') if torch.cuda.is_available() else input_ids.to('cpu')
        generated_text = model.generate(input_ids, 
                                        max_length=256,   
                                        temperature=0.7,  
                                        num_return_sequences=5)
        
        # decode and post-process output into a readable format
        query = tokenizer.decode(generated_text[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(' ','').split('\n')

        # return results as a string
        result = ''
        
        for q in query:
            result += f'{q}\n'
            
        return result

# test_function_code --------------------

def test_generate_queries():
    '''
    Test the generate_queries function.
    '''
    document1 = 'This is a test document.'
    document2 = 'Another test document.'
    document3 = 123

    assert isinstance(generate_queries(document1), str)
    assert isinstance(generate_queries(document2), str)
    try:
        generate_queries(document3)
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError not raised for non-string input.')

    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_queries()