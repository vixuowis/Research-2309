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
    
    # load the model and tokenizer from Hugging Face
    try:
        model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    except requests.exceptions.ChunkedEncodingError:
        raise ConnectionError("There was a connection error while trying to download the model.")

    # generate the summarization
    text = input_text if len(input_text) < 1024 else input_text[:1024]
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    # remove the ">>" that is generated at the beginning of the summary
    summary = output[0][3:]

    return summary

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