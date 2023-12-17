# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def converse_with_user(input_text):
    """
    This function initializes the conversational pipeline with the 'hyunwoongko/blenderbot-9B' model
    and generates a response from the chatbot given the user's input text.
    
    :param input_text: A string containing the user's message to the chatbot.
    :return: A string containing the chatbot's generated response.
    """
    # Initialize the conversational pipeline with the specified model
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    
    # Generate and return the chatbot's response to the input text
    response = chatbot(input_text)
    return response[0]['generated_text']

# test_function_code --------------------

def test_converse_with_user():
    print("Testing started.")

    # 测试用例 1ï¼问候语
    print("Testing case [1/3] started.")
    greeting_response = converse_with_user("Hello, how are you?")
    assert isinstance(greeting_response, str), f"Test case [1/3] failed: The response should be a string but got {type(greeting_response)}."

    # 测试用例 2ï¼通用问题
    print("Testing case [2/3] started.")
    question_response = converse_with_user("What's the weather like today?")
    assert isinstance(question_response, str), f"Test case [2/3] failed: The response should be a string but got {type(question_response)}."

    # 测试用例 3ï¼随机输入测试模型的鲁梦性
    print("Testing case [3/3] started.")
    random_input_response = converse_with_user("sjdksla qwoiu?")
    assert isinstance(random_input_response, str), f"Test case [3/3] failed: The response should be a string but got {type(random_input_response)}."
    
    print("Testing finished.")

# 运行测试函数
test_converse_with_user()