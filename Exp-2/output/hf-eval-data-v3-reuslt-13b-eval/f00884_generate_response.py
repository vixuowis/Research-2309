# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_response(user_input: str) -> str:
    """
    Generate a response to the user input using the BlenderbotForConditionalGeneration model.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The model's response.

    Raises:
        OSError: If there is a problem with the model loading or the disk quota is exceeded.
    """

    # Load the BlenderbotForConditionalGeneration model from Hugging Face Hub. This will automatically download it if
    # it's not already present on your machine.

    try:
        model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-90M")
        tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-90M")
    except Exception as e:
        print(f"Error loading model for BlenderBot, exception: {e}")
        raise OSError("Disk quota exceeded")

    # Tokenize the user input and add a batch dimension to it. The BlenderbotForConditionalGeneration model expects
    # inputs in batches, so we need to create a batch of 1 for this example.

    tokenized_user_input = tokenizer([user_input], max_length=500, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokenized_user_input["input_ids"]
    attention_mask = tokenized_user_input["attention_masks"]
    batched_input = {
        "input_ids": input_ids.reshape(1, -1),  # shape (batch_size, sequence_length).
        "attention_mask": attention_mask.reshape(1, -1)  # shape (batch_size, sequence_length).
    }

    # Generate the response using our model and tokenizer. The BlenderbotForConditionalGeneration model has been
    # trained to generate responses of max length 50 for inputs that are max length 500.

    try:
        generated_output = model.generate(**batched_input, max_length=50)
    except Exception as e:
        print(f"Error generating response from BlenderBot, exception: {e}")
        raise OSError("Disk quota exceeded")

    # Convert the tokenized output to a string and return it. The tokenizer will also add special tokens into the
    # response. We'll remove

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    user_input = 'What are the benefits of regular exercise?'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    user_input = 'Tell me a joke.'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    user_input = 'What is the weather like today?'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()