# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_conference(text):
    # Using the Pegasus model for summarization
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    # Summarize the provided text
    summary = summarizer(text, truncation=True)

    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_conference():
    print("Testing summarize_conference started.")
    # Test input: example conference description
    conference_description = "The conference on climate change and health covered heat-related illnesses, vector-borne diseases, and air pollution. Experts advocated for investment in climate change mitigation for health protection."
    # Expected output: Summary of the conference
    expected_summary = "The World Health Organization's conference on climate change and human health examined heat-related illnesses, vector-borne diseases and air pollution. Experts urged governments to invest in mitigating the negative impacts of climate change on health."
    print("Testing summarize_conference.")
    assert summarize_conference(conference_description) == expected_summary, "Test failed: The summary does not match the expected output."
    print("Testing summarize_conference finished successfully.")

# Run the test function
if __name__ == '__main__':
    test_summarize_conference()