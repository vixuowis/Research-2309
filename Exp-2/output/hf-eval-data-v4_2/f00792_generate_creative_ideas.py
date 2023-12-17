# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_creative_ideas(prompt_text, max_length=50, num_ideas=5):
    """Generates creative ideas for a given prompt text using a pre-trained model.

    Args:
        prompt_text (str): The initial text to spark the generation.
        max_length (int): The maximum length of each generated idea.
        num_ideas (int): The number of creative ideas to generate.

    Returns:
        list: A list of generated ideas based on the prompt.

    Raises:
        RuntimeError: If generation fails due to model or input issues.
    """
    try:
        generator = pipeline('text-generation', model='distilgpt2')
        set_seed(42)
        ideas = generator(prompt_text, max_length=max_length, num_return_sequences=num_ideas)
        return [idea['generated_text'] for idea in ideas]
    except Exception as e:
        raise RuntimeError('Failed to generate creative ideas: ' + str(e))

# test_function_code --------------------

def test_generate_creative_ideas():
    print("Testing started.")

    print("Testing case [1/1] started.")
    generated_ideas = generate_creative_ideas("Once upon a time,", 50, 5)
    assert isinstance(generated_ideas, list) and len(generated_ideas) == 5, f"Test case [1/1] failed: Expected 5 generated ideas, got {len(generated_ideas)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_creative_ideas()