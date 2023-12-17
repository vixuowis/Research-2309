# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story_start(prompt):
    """
    Generates a story start based on the provided prompt using text generation model.

    Args:
        prompt: A string containing the initial text to start the story from.

    Returns:
        A dictionary containing the generated story start text.

    Raises:
        ValueError: If the prompt is empty or None.
    """
    if not prompt:
        raise ValueError("The prompt is empty or None.")

    text_generator = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    story_start = text_generator(prompt, max_length=50, num_return_sequences=1)
    return story_start[0]['generated_text']

# test_function_code --------------------

def test_generate_story_start():
    print("Testing started.")
    
    # 测试用例 1：使用非空的提示生成故事开头
    print("Testing case [1/1] started.")
    prompt = "A brave knight and a fearsome dragon"
    story = generate_story_start(prompt)
    
    assert story.startswith(prompt), f"Test case [1/1] failed: Generated story does not start with the prompt."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_story_start()