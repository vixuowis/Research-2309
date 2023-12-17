# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_executive_meeting_notes(meeting_notes):
    """
    Summarize the meeting notes of an executive without revealing too much detail.

    Parameters:
    meeting_notes (str): The text of the executive's meeting notes.

    Returns:
    str: A summary of the meeting notes.
    """
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(meeting_notes, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_executive_meeting_notes():
    print("Testing summarize_executive_meeting_notes started.")
    sample_notes = "During the meeting, discussions were held about the upcoming product launch, marketing strategies, and budget allocations. It is crucial that the timelines are strictly followed and the quality of the product is not compromised."
    print("Testing case started.")
    summary = summarize_executive_meeting_notes(sample_notes)
    assert summary, f"Test case failed: Summary is empty"
    print(f"Test case passed: Summary - {summary}")
    print("Testing finished.")

test_summarize_executive_meeting_notes()