# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import requests

# function_code --------------------

def summarize_text(input_text: str) -> str:
    """
    Summarizes a given text using the Pegasus model from Hugging Face Transformers.

    Args:
        input_text (str): The text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        requests.exceptions.ChunkedEncodingError: If there is a connection error while downloading the model.
    """
    
    # Loading tokenizer and model
    try:
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        
    except requests.exceptions.ChunkedEncodingError as e:
        print("There was a connection error while downloading the model, please check your internet connection.")
        exit() 
    
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

    # Tokenize the text and generate tokens of batch size 1
    input_ids = tokenizer(input_text, truncation=True, padding='longest', return_tensors="pt").input_ids
    
    if len(input_ids) == 0:
        print("The input text is empty.")
        exit() 

    # Generate the summarized text
    summary = model.generate(input_ids, max_length=256, length_penalty=1.0, num_beams=4)

    return tokenizer.batch_decode(summary, skip_special_tokens=True)[0]

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the summarize_text function with some example texts.
    """
    input_text1 = 'A new study suggests that eating chocolate at least once a week can lead to better cognition.'
    input_text2 = 'The study, published in the journal Appetite, analyzed data from over 900 adults and found that individuals who consumed chocolate at least once a week performed better on cognitive tests than those who consumed chocolate less frequently.'
    input_text3 = 'Researchers believe that the beneficial effects of chocolate on cognition may be due to the presence of flavonoids, which have been shown to be antioxidant-rich and to improve brain blood flow.'
    assert len(summarize_text(input_text1)) > 0
    assert len(summarize_text(input_text2)) > 0
    assert len(summarize_text(input_text3)) > 0
    return 'All Tests Passed'


# call_test_function_code --------------------

test_summarize_text()