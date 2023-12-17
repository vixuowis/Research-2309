# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def create_conversational_bot(model_name: str):
    """
    Create a conversational bot using the specified model.

    Parameters:
    model_name (str): The name of the model to use.

    Returns:
    callable: A function to converse with the bot.
    """
    chatbot = pipeline('conversational', model=model_name)
    return chatbot


# test_function_code --------------------

def test_conversational_bot():
    print("Testing the conversational bot creation.")
    bot = create_conversational_bot('mywateriswet/ShuanBot')
    assert callable(bot), "The bot should be callable."
    response = bot("What is your name?")
    assert isinstance(response, list), "The bot should return a list of responses."
    assert len(response) > 0, "The bot should return at least one response."
    print("Testing completed successfully.")

test_conversational_bot()
