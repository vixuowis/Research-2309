# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

# function_code --------------------

def chat_with_home_appliances(input_message):
    # Load the tokenizer and model from Hugging Face's Transformers
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-90M')
    model = AutoModelForCausalLM.from_pretrained('facebook/blenderbot-90M')

    # Encode the user input and generate a response
    tokenized_input = tokenizer.encode(input_message + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(tokenized_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, tokenized_input.shape[-1]:][0], skip_special_tokens=True)

    # Return the response which can be utilized by the home appliance systems
    return response

# test_function_code --------------------

def test_chat_with_home_appliances():
    print('Testing chat_with_home_appliances function')

    # Test case: User asks to turn on a light
    input_message = 'Turn on the living room lights.'
    response = chat_with_home_appliances(input_message)
    assert response, 'Test case failed: No response generated for turning on lights.'

    # Test case: User asks to turn off the TV
    input_message = 'Can you turn off the TV please?'
    response = chat_with_home_appliances(input_message)
    assert response, 'Test case failed: No response generated for turning off the TV.'

    print('All tests passed for chat_with_home_appliances function')