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
    
    model_name = 'sshleifer/distilbart-cnn-12-6'
    
    model_dir = '/opt/ml/input/artifacts/model/' if run_locally else sagemaker_session.download_model(job_name, f'{model_name}')
    tokenizer = BartTokenizer.from_pretrained(f'{model_dir}/tokenizer') 
    model = BartForConditionalGeneration.from_pretrained(f'{model_dir}/model/')
    
    input_ids = tokenizer([input_text], return_tensors="pt", padding=True).input_ids # add batch dimension (see https://huggingface.co/transformers/main_classes/tokenizer.html)
    summary_ids = model.generate(input_ids)[0]
    
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

# ----------------------------------

# main_code ------------------------

if __name__ == '__main__':
    
    run_locally = False # set this flag to true if running code locally and not in an AWS SageMaker environment
    
    if run_locally: 
        
        text = "Summarize this article, please!"
                     
        model_name = 'sshleifer/distilbart-cnn-12-6'
        
        tokenizer = BartTokenizer.from_pretrained(model_name) # add additional arguments as required by the model
        model = BartForConditionalGeneration.from_pretrained(model_name, force_download=True) 
    
        input_ids = tokenizer([text], return_tensors="pt", padding='max_length', truncation=True, max_length=1024).input_ids # add batch dimension (see https://huggingface.co/transformers/main_classes/tokenizer.html)
        summary_ids = model.generate(input_ids)[0]
        
        print('The summarized text is:')
        print(

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