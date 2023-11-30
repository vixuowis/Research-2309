# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_marketing_content(prompt: str) -> str:
    """
    Generate marketing content using the OPT pre-trained transformer 'facebook/opt-125m'.

    Args:
        prompt (str): The initial prompt to feed to the text generation model.

    Returns:
        str: The generated marketing content.

    Raises:
        OSError: If there is a problem with the disk quota.
    """    
        
    # Create and configure pipeline for generation --------------------
    try:
        set_seed(42)
        model = "facebook/opt-125m"
        tokenizer = "facebook/opt-125m"
        summarization_pipeline = pipeline("summarization", 
                                          model=model, 
                                          tokenizer=tokenizer)
    except OSError as err:
        print(f'Error code:\t{err.errno}\n\nError message:\t{err.strerror}')
        
    # Generate marketing content --------------------
    
    try:
        prompt = summarization_pipeline(prompt)[0]['summary_text']
    except IndexError as err:
        print(f'Error code:\t\t\t{err.args[1]}\nError message:\t\t{err.args[0]}')
        
    return prompt

# test_function_code --------------------

def test_generate_marketing_content():
    """
    Test the generate_marketing_content function.
    """
    prompt = 'Introducing our new line of eco-friendly kitchenware:'
    generated_content = generate_marketing_content(prompt)
    assert isinstance(generated_content, str)
    assert len(generated_content) > 0
    print('All Tests Passed')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_generate_marketing_content()