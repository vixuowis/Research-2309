# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(text: str) -> str:
    '''
    Summarizes the input text using the PEGASUS model from Hugging Face Transformers.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    '''
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_text():
    '''
    Tests the summarize_text function.
    '''
    text1 = 'Over the past week, the World Health Organization held a conference discussing the impacts of climate change on human health. The conference brought together leading experts from around the world to examine the current problems affecting people\'s health due to changing environmental conditions. The topics of discussion included increased occurrence of heat-related illnesses, heightened rates of vector-borne diseases, and the growing problem of air pollution. The conference concluded with a call to action for governments and organizations to invest in mitigating and adapting to the negative consequences of climate change for the sake of public health.'
    assert len(summarize_text(text1)) > 0

    text2 = 'The World Health Organization is a specialized agency of the United Nations responsible for international public health. The WHO Constitution, which establishes the agency\'s governing structure and principles, states its main objective as ensuring the attainment by all peoples of the highest possible level of health.'
    assert len(summarize_text(text2)) > 0

    text3 = 'Climate change is a long-term shift in weather conditions identified by changes in temperature, precipitation, winds, and other indicators. Climate change can involve both changes in average conditions and changes in variability, including, for example, extreme events.'
    assert len(summarize_text(text3)) > 0

    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()