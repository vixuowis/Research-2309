# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(description: str) -> str:
    '''
    Generate code snippets based on natural language descriptions.

    Args:
        description (str): The natural language description.

    Returns:
        str: The generated code snippet.
    '''

    
    model_name = 'microsoft/DialoGPT-medium'   # or 'gpt2', 'openai-gpt' and more

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    
    def _get_code_line(text):  # generate a line of code from the given description
        input_context = 'if True:\n' if text[0] == '@' else ''
        return f"{input_context}{text.replace(' ', '')}\n"
    
    encoded_prompt = tokenizer.encode(description, add_special_tokens=False, return_tensors='pt')
    encoded_prompt = encoded_prompt.to(device) if device.type != 'cpu' else encoded_prompt
    output_sequences = model.generate(
        input_ids=encoded_prompt, 
        max_length=50, 
        temperature=1., 
        top_k=50,
        top_p=1.0
    )
    
    generated_sequence = output_sequences[0]
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    code_lines = [_get_code_line(line) for line in text.split('\n') if line and (not '|' in line)]
    
    return '\n'.join(code_lines)

# test_function_code --------------------

def test_generate_code():
    '''
    Test the generate_code function.
    '''
    description1 = 'Write a Python function to calculate the factorial of a number.'
    description2 = 'Write a Python function to sort a list of numbers in ascending order.'
    description3 = 'Write a Python function to reverse a string.'
    assert isinstance(generate_code(description1), str)
    assert isinstance(generate_code(description2), str)
    assert isinstance(generate_code(description3), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_code()