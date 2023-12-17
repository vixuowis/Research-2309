# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_article(text):
    '''
    Summarize the given article text using BigBird Pegasus model.

    Args:
        text (str): The article text to summarize.

    Returns:
        str: The summarized article text.
    '''
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    inputs = tokenizer(text, return_tensors='pt')
    prediction = model.generate(**inputs)
    summary = tokenizer.batch_decode(prediction, skip_special_tokens=True)[0]
    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing summarization function started.")
    sample_text = "This is a long article text that needs to be summarized to understand the main points quickly without having to read the entire content."

    # Test case: Summarize article text
    print("Testing summarization started.")
    summary = summarize_article(sample_text)
    assert isinstance(summary, str), f"Summarization failed, expected a string output, got: {type(summary)}"
    assert len(summary) < len(sample_text), "Summarization failed, summary is not shorter than the original text."
    assert "main points" in summary or "summarized" in summary, "Summarization failed, expected keywords are missing in the summary."
    print("Testing finished successfully.")

# Run the test function
test_summarize_article()