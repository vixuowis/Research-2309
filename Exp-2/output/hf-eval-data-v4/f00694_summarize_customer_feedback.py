# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_customer_feedback(feedback):
    """
    Generate a summary for the given customer feedback using the Hugging Face Transformers pipeline.

    :param feedback: str - A lengthy string of customer feedback.
    :return: str - The summary of the customer feedback.
    """
    summarizer = pipeline('summarization', model='philschmid/bart-large-cnn-samsum')
    summary = summarizer(feedback)
    return summary[0]['summary_text']


# test_function_code --------------------

def test_summarize_customer_feedback():
    print("Testing summarize_customer_feedback function.")

    # Sample customer feedback
    feedback = (
        "The customer support service was excellent. All our concerns were attended to promptly "
        "by the friendly and knowledgeable staff. However, the user interface could use some "
        "improvement as navigating the platform can be challenging."
    )

    # Expected summary based on the model's performance
    expected_summary = "The customer support service was excellent, but the user interface could use improvement."

    # Generate the summary
    summary = summarize_customer_feedback(feedback)

    # Test if the generated summary matches the expected summary
    assert summary == expected_summary, (
        f"Test failed: Expected summary '{expected_summary}', but got '{summary}'."
    )

    print("Test passed: The function 'summarize_customer_feedback' works as expected.")

# Run the test
if __name__ == '__main__':
    test_summarize_customer_feedback()
