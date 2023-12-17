# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

def generate_questions(paragraph, model_name='castorini/doc2query-t5-base-msmarco', max_length=100):
    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Prepare the text input for the model
    input_text = f'generate questions: {paragraph}'
    inputs = tokenizer.encode(input_text, return_tensors='pt', padding=True)

    # Generate questions from the paragraph
    outputs = model.generate(inputs, max_length=max_length)
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return questions

# test_function_code --------------------

def test_generate_questions():
    print("Testing generate_questions function.")

    # Test case: Test with a sample paragraph
    sample_paragraph = 'Transformers is an open-source library for natural language processing.'
    expected_output = 'What is Transformers?'
    print("Testing with sample paragraph.")
    questions = generate_questions(sample_paragraph)
    print(f'Generated questions: {questions}')
    assert expected_output in questions, f"Test failed: Expected question not found in generated questions."

    print("All tests passed.")

# Run the test function
test_generate_questions()