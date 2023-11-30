# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_response(user_input: str, chat_history: torch.Tensor = None):
    """
    Generate a response for the given user input using DialoGPT-medium model.

    Args:
        user_input (str): The input message from user.
        chat_history (torch.Tensor, optional): The chat history. Defaults to None.

    Returns:
        Tuple[str, torch.Tensor]: The generated response and the updated chat history.
    """    

    # Set up tokenizer & model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium").to(device)

    # Create input sequence and encode it for the chatbot
    if user_input is None or (isinstance(user_input, str) and user_input.strip() == ""):
        raise ValueError("Input should not be empty.")
    
    if isinstance(chat_history, torch.Tensor):
        history = chat_history
    else:
        history = None
    
    while True:
        
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)  # set the input sequence
        
        if history is not None:
            chat_history = torch.cat([chat_history, input_ids], dim=-1)
            input_ids = chat_history
            
        # Generate chatbot response based on input and chat history
        chat_history = model.generate(input_ids, max_length=200, pad_token_id=50256)
        
        bot_response = tokenizer.decode(chat_history[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            
        if bot_response.strip() != "":
            break
    
    return (bot_response, chat_history)

# test_function_code --------------------

def test_generate_response():
    user_input = 'Hello, how are you?'
    response, chat_history = generate_response(user_input)
    assert isinstance(response, str)
    assert isinstance(chat_history, torch.Tensor)
    user_input = 'What is your name?'
    response, chat_history = generate_response(user_input, chat_history)
    assert isinstance(response, str)
    assert isinstance(chat_history, torch.Tensor)
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_response())