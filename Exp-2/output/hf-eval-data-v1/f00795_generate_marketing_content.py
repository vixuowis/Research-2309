from transformers import pipeline, set_seed


def generate_marketing_content(prompt: str, max_length: int = 100, do_sample: bool = True) -> str:
    """
    Generate marketing content for a given prompt using the OPT pre-trained transformer 'facebook/opt-125m'.

    Args:
        prompt (str): The initial prompt to start the text generation.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
        do_sample (bool, optional): Whether or not to use sampling in text generation. Defaults to True.

    Returns:
        str: The generated marketing content.
    """
    set_seed(42)
    generator = pipeline('text-generation', model='facebook/opt-125m')
    generated_content = generator(prompt, max_length=max_length, do_sample=do_sample)[0]['generated_text']
    return generated_content