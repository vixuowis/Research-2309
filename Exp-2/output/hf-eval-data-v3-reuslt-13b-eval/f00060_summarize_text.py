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
    # Load the model and tokenizer from Huggingface Hub.
    print('Loading models...')
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    
    # Tokenize the text with truncation and padding.
    print('Tokenizing input...')
    inputs = tokenizer(text, max_length=1024, return_tensors="pt",truncation=True,padding='max_length')

    # Generate summarization using beam search decoding with alpha parameter of 0.95.
    print('Generating summary...')
    outputs = model.generate(
        inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=3, max_length=256, min_length=100, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True
    )
    
    # De-tokenize the output to get human readable text and return it.
    print('Converting to string...')
    return tokenizer.batch_decode(outputs)[0]

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