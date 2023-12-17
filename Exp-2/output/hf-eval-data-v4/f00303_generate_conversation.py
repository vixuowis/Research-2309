# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_conversation(input_message, character_persona, dialogue_history=""):
    '''
    Generate a conversational response using a pre-trained model.

    Parameters:
    input_message (str): The input message from the user.
    character_persona (str): The persona of the chatbot.
    dialogue_history (str): The dialogue history of the conversation.

    Returns:
    str: The generated response from the chatbot.
    '''
    chatbot = pipeline('text-generation', 'PygmalionAI/pygmalion-1.3b')
    input_prompt = f"{character_persona}\n{dialogue_history}You: {input_message}\n"
    response = chatbot(input_prompt)
    return response[0]['generated_text']

# test_function_code --------------------

def test_generate_conversation():
    print("Testing generate_conversation function.")

    character_persona = "CompanyBot's Persona: I am a helpful chatbot designed to answer questions about our products and services."
    test_input = "What products do you offer?"
    dialog_history = ""

    # Test case 1
    print("Testing case [1/1] started.")
    response = generate_conversation(test_input, character_persona, dialog_history)
    assert 'CompanyBot:' in response, f"Test case [1/1] failed: Expected the chatbot's persona to be included in the response."
    print("Testing case [1/1] finished.")
    print("Testing finished.")

# Run test function
test_generate_conversation()