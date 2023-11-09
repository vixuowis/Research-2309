# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(input_text):
    """
    Summarize a given text using PegasusForConditionalGeneration model from Hugging Face Transformers.

    Args:
        input_text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_text():
    """
    Test the summarize_text function.
    """
    input_text = 'A new study suggests that eating chocolate at least once a week can lead to better cognition. The study, published in the journal Appetite, analyzed data from over 900 adults and found that individuals who consumed chocolate at least once a week performed better on cognitive tests than those who consumed chocolate less frequently. Researchers believe that the beneficial effects of chocolate on cognition may be due to the presence of flavonoids, which have been shown to be antioxidant-rich and to improve brain blood flow.'
    summary = summarize_text(input_text)
    assert isinstance(summary, str) and len(summary) < len(input_text)

# call_test_function_code --------------------

test_summarize_text()