# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_and_generate_question(input_text):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')
    model = AutoModelForSeq2SeqLM.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')

    # Tokenize input text for the model
    tokenized_input = tokenizer(input_text, return_tensors="pt")

    # Generate summary and question
    output = model.generate(**tokenized_input)
    
    # Decode the output token ids to a text string
    summary_and_question = tokenizer.decode(output[0], skip_special_tokens=True)

    return summary_and_question

# test_function_code --------------------

def test_summarize_and_generate_question():
    print("Testing started.")
    # Sample data for testing
    sample_data = "This is an example of input text to be summarized and transformed into an open-ended question."

    # Testing the function
    result = summarize_and_generate_question(sample_data)

    # Test: Check if the result is a non-empty string
    assert result, "The result of summarization is empty."
    
    print("Test case passed: The result of summarization is non-empty.")
    print("Testing finished.")