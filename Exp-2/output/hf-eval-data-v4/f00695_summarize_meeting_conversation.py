# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_meeting_conversation(conversation):
    # Load the text summarization model
    summarizer = pipeline('summarization', model='philschmid/distilbart-cnn-12-6-samsum')

    # Generate a summary of the team meeting conversation
    summary = summarizer(conversation)

    # Return the summary text
    return summary[0]['summary_text']

# Example usage
# conversation_text = "Anna: In today's meeting, we discussed increasing marketing budget. Tom: I suggested allocating more funds to..."
# print(summarize_meeting_conversation(conversation_text))

# test_function_code --------------------

def test_summarize_meeting_conversation():
    print("Testing summarize_meeting_conversation function.")

    # A test conversation
    test_conversation = "Anna: In today's meeting, we discussed increasing marketing budget. Tom: I suggested allocating more funds to..."

    # Expected summary (this is not an actual expected result, just for illustrative purposes)
    expected_summary = "The team discussed increasing the marketing budget, focusing on social media campaigns and SEO, and agreed to invest in content creation."

    # Run the function
    summary = summarize_meeting_conversation(test_conversation)

    # Check if the generated summary is not empty
    assert summary, "The summary is empty."

    print("Test passed. Function is working as expected.")

# Run the test
# test_summarize_meeting_conversation()