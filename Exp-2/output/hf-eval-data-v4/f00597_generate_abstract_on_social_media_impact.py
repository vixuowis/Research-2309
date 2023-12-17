# requirements_file --------------------

!pip install -U torch tokenizers transformers

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def generate_abstract_on_social_media_impact(text):
    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')

    # Encode input text
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    # Use summarization prefix for T5 to understand the task
    decoder_input_ids = tokenizer('summarize: ', return_tensors='pt').input_ids

    # Generate the output abstract by running the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    # Decode and return the generated abstract
    abstract = tokenizer.decode(outputs.last_hidden_state.squeeze(), skip_special_tokens=True)
    return abstract

# test_function_code --------------------

def test_generate_abstract_on_social_media_impact():
    print("Testing generate_abstract_on_social_media_impact function.")
    sample_text = "Social media platforms have rapidly become integral to many people's lives, but their effect on mental health is a growing concern."

    # Expected output is an abstract summarizing the impact of social media on mental health

    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    abstract = generate_abstract_on_social_media_impact(sample_text)
    assert isinstance(abstract, str), "Test case [1/1] failed: The function should return a string."
    print("Testing case [1/1] successful.")
    print("Testing finished.")

# Run the test function
test_generate_abstract_on_social_media_impact()