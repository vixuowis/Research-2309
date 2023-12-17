# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import LEDForConditionalGeneration, LEDTokenizer

# function_code --------------------

def summarize_dialogue(input_text):
    # Load the LED model for dialogue summarization from Hugging Face Transformers
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    tokenizer = LEDTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')

    # Tokenize the input text to prepare it for the model
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')

    # Generate summary tokens
    summary_ids = model.generate(input_tokens)

    # Decode the generated tokens into a summary string
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_dialogue():
    print("Testing summarize_dialogue function.")
    # Example of a lengthy dialogue
    dialogue_example = "... (Insert lengthy dialogue here) ..."

    # Generate summary for the dialogue
    summary = summarize_dialogue(dialogue_example)

    # Check that the summary is not empty and is shorter than the original dialogue
    assert summary, "The summary should not be empty."
    assert len(summary) < len(dialogue_example), "The summary should be shorter than the original dialogue."

    print("Summary:", summary)
    print("Test passed.")

# Run the test function
test_summarize_dialogue()