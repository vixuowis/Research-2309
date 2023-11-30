# function_import --------------------

import requests
from transformers import BartTokenizer, BartForConditionalGeneration

# function_code --------------------

def summarize_text(input_text: str) -> str:
    """
    Summarize a given text using the pre-trained model 'sshleifer/distilbart-cnn-12-6'.

    Args:
        input_text (str): The text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        requests.exceptions.ChunkedEncodingError: If there is a connection error while downloading the model.
    """

    try:
         # Load the tokenizer and model from HuggingFace Hub
         tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
         model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6').to("cpu")

    except requests.exceptions.ChunkedEncodingError:
        raise requests.exceptions.ChunkedEncodingError(
            "Could not download the model 'sshleifer/distilbart-cnn-12-6'.\n" +
             "Please try again."
         )
    
    # Preprocessing the text to remove the unnecessary characters and lowercase the words.
    input_text = str(input_text).replace('\"','').replace('.','.\n ').lower().strip()

    max_length = 128
    min_length = 35

    # Tokenize the input text.
    encoded_input = tokenizer([input_text], max_length=max_length, truncation=True, return_tensors="pt")
    encoded_inputs = encoded_input["input_ids"]

    # Perform summarization of the text using HuggingFace model.
    output = model.generate(encoded_inputs, min_length=min_length, max_length=max_length, length_penalty=2)
    
    summary_text = tokenizer.batch_decode(output)[0]

    return summary_text

# test_function_code --------------------

def test_summarize_text():
    """Test the function summarize_text."""
    input_text1 = 'This is a long article about the history of the world. It covers many different topics and periods.'
    input_text2 = 'This is another long article, this time about the future of technology. It discusses many potential advancements and challenges.'
    assert isinstance(summarize_text(input_text1), str)
    assert isinstance(summarize_text(input_text2), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_summarize_text()