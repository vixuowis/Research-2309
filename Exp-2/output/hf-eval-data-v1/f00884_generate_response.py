from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer


def generate_response(user_input: str) -> str:
    """
    Generate a response to the user input using the pre-trained Blenderbot model.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The model's response.
    """
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-3B')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-3B')

    inputs = tokenizer([user_input], return_tensors='pt')
    outputs = model.generate(**inputs)
    reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return reply