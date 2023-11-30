# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(text):
    """
    Generate executable code based on the input prompt using Hugging Face Transformers.

    Args:
        text (str): The input prompt in English.

    Returns:
        str: The generated executable code.
    """
    
    # tokenizer and model for transformer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
    model = AutoModelForCausalLM.from_pretrained("https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2", return_dict=True)  # pylint: disable=line-too-long
    
    inputs = tokenizer(text, truncation="longest_first", max_length=1024, return_tensors="pt") 
    outputs = model.generate(inputs["input_ids"], do_sample=True, top_k=50, max_length=1023)   # pylint: disable=line-too-long
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# test_function_code --------------------

def test_generate_code():
    """
    Test the function generate_code.
    """
    assert generate_code('Create a simple loading spinner for maintenance.') is not None
    assert generate_code('Create a function to add two numbers.') is not None
    assert generate_code('Create a function to calculate the factorial of a number.') is not None
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_code()