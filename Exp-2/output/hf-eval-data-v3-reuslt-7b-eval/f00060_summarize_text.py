# function_import --------------------

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_text(text):
    """
    Summarizes a given long text using BigBird Pegasus model.

    Args:
        text (str): The long text to be summarized.

    Returns:
        str: The summarized text.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-arxiv')
    
    inputs  = tokenizer([text], max_length=2048, truncation=True)
    input_ids = inputs["input_ids"]
    input_mask = [1] * len(input_ids[0])
    output_ids = model.generate(input_ids, attention_mask=input_mask, num_beams=4, length_penalty=2.0, early_stopping=True, max_length=5)
    
    return tokenizer.decode(output_ids[0], skip_special_tokens = True).capitalize()

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the summarize_text function with some test cases.
    """
    test_text1 = 'This is a long text that needs to be summarized. It contains many details that are not necessary for understanding the main idea.'
    test_text2 = 'Another long text that needs summarization. It also contains many unnecessary details.'
    assert len(summarize_text(test_text1)) < len(test_text1)
    assert len(summarize_text(test_text2)) < len(test_text2)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_summarize_text()