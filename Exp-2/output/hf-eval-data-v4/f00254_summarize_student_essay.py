# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BartTokenizer, BartModel

# function_code --------------------

def summarize_student_essay(essay_text):
    """
    Summarizes a student's essay using the BART model.

    Parameters:
    essay_text (str): The text of the student's essay to be summarized.

    Returns:
    str: A summary of the essay.
    """
    # Load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')

    # Tokenize the essay text
    inputs = tokenizer(essay_text, max_length=1024, return_tensors='pt', truncation=True)

    # Generate a summary with the model
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# test_function_code --------------------

def test_summarize_student_essay():
    print("Testing started.")
    
    # Test case 1: A short essay
    print("Testing case [1/2] started.")
    short_essay = "Once upon a time, a little girl named Alice was wandering in the forest when she stumbled upon a curious little rabbit hole. After falling down the hole, Alice finds herself in a whimsical world full of strange creatures and wonders."
    summary1 = summarize_student_essay(short_essay)
    assert len(summary1) < len(short_essay), f"Test case [1/2] failed: Summary is not shorter than the original essay."

    # Test case 2: A long essay
    print("Testing case [2/2] started.")
    long_essay = "In the midst of the industrial revolution, many workers found themselves... (Add long essay text here)"
    summary2 = summarize_student_essay(long_essay)
    assert len(summary2.split()) <= 20, f"Test case [2/2] failed: Summary length exceeds 20 words."

    print("Testing finished.")

# Run the test function
test_summarize_student_essay()