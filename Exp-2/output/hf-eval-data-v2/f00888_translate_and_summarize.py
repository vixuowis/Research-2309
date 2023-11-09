# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_and_summarize(input_text: str, model_name: str = 'google/flan-t5-large') -> str:
    """
    Translates English text to German and summarizes a given paragraph using the FLAN-T5 large model.

    Args:
        input_text (str): The text to be translated and summarized.
        model_name (str, optional): The name of the model to be used. Defaults to 'google/flan-t5-large'.

    Returns:
        str: The translated and summarized text.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

# test_function_code --------------------

def test_translate_and_summarize():
    """
    Tests the translate_and_summarize function.
    """
    input_text = 'I have a doctor\'s appointment tomorrow morning.'
    expected_output = 'Ich habe morgen früh einen Arzttermin.'
    assert translate_and_summarize(input_text).startswith(expected_output)

    input_text = 'Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn patterns from data without being explicitly programmed. The field has seen tremendous growth in recent years, driven by advances in computational power, the abundance of data, and improvements in algorithms. There are many types of machine learning algorithms, including supervised learning, unsupervised learning, reinforcement learning, and deep learning. Applications of machine learning are diverse, ranging from image and speech recognition to financial trading and recommendation systems.'
    expected_output = 'Machine learning, a subset of artificial intelligence, develops algorithms to learn patterns without explicit programming. Driven by computational advancement, abundant data, and algorithmic improvements, it includes supervised, unsupervised, reinforcement, and deep learning algorithms. Applications span from image and speech recognition to financial trading and recommendation systems.'
    assert translate_and_summarize(input_text).startswith(expected_output)

# call_test_function_code --------------------

test_translate_and_summarize()