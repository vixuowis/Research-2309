# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_nature_poem(prompt):
    """
    Generate a poem about nature based on a given text prompt using the tiny-gpt2 model from Hugging Face Transformers.

    Parameters:
    prompt (str): A string containing the initial text prompt for poem generation.

    Returns:
    str: A generated poem.
    """
    nlp = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    result = nlp(prompt, max_length=100, clean_up_tokenization_spaces=True)
    poem = result[0]['generated_text']
    return poem.strip()

# test_function_code --------------------

def test_generate_nature_poem():
    print("Testing generate_nature_poem function.")
    test_prompt = "Once upon a time, in a land of greenery and beauty,"
    
    # 测试用例 1：检查是否返回非空字符串
    print("Testing case [1/3] started.")
    poem_result = generate_nature_poem(test_prompt)
    assert poem_result, "Test case [1/3] failed: The function returned an empty string."

    # 测试用例 2：检查返回字符串开始于提示文本
    print("Testing case [2/3] started.")
    assert poem_result.startswith(test_prompt), f"Test case [2/3] failed: The poem does not start with the prompt. Returned poem: {poem_result}"

    # 测试用例 3：检查返回字符串长度大于提示文本
    print("Testing case [3/3] started.")
    assert len(poem_result) > len(test_prompt), f"Test case [3/3] failed: The poem is not longer than the prompt. Returned poem: {poem_result}"
    
    print("Testing finished.")

# 运行test_generate_nature_poem()
test_generate_nature_poem()