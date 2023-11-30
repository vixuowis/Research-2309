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

    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6').to("cuda")

    input_text = input_text.replace('\n', ' ')
        
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
    
    output_summary = model.generate(
        input_ids.to("cuda"), 
        max_length=250, 
        num_beams=6, 
        early_stopping=True
    )
    
    summary_text = tokenizer.decode(output_summary[0], skip_special_tokens=True)
        
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