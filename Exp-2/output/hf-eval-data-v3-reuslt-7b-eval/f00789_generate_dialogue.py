# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_dialogue(user_input):
    """
    Generate a dialogue response using DialoGPT-large model.

    Args:
        user_input (str): The user's input to which the chatbot should respond.

    Returns:
        str: The chatbot's response.

    Raises:
        OSError: If there is an issue with loading the pre-trained model or tokenizer.
    """

    try:
        
        # Create a list to store generated text in
        generated_text = []
        
        # Load our pre-trained model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large") 
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", use_fast=True)
        
        # Encode the input text as a PyTorch tensor
        user = torch.tensor(tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')).to('cuda')

        # Generate output tokens using model generated tokens as the input
        chatbot_output = model.generate(user, max_length=150)  
        
        # Decode to text and add to list of responses
        for i in range(chatbot_output.shape[1]):
            generated_text.append(tokenizer.decode(chatbot_output[:, i], skip_special_tokens=True)) 
            
        return generated_text[-1]
        
    except OSError as e:
        return f"Error: {e}"

# test_function_code --------------------

def test_generate_dialogue():
    """
    Test the generate_dialogue function.
    """
    test_input = 'How do I search for scientific papers?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'What is the weather like today?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'Tell me a joke.'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_dialogue()