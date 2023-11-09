from transformers import pipeline


def generate_game_setting(initial_text):
    """
    Generate a game setting using a pre-trained text generation model.

    Args:
        initial_text (str): The initial text to feed into the text generation model.

    Returns:
        str: The generated text that can serve as a game setting.
    """
    model = pipeline('text-generation', model='bigscience/bloom-7b1')
    result = model(initial_text)
    return result[0]['generated_text']