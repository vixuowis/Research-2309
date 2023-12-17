# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_creative_ideas(prompt, max_length=50, num_return_sequences=5):
    set_seed(42)
    generator = pipeline('text-generation', model='distilgpt2')
    ideas = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return ideas

# test_function_code --------------------

def test_generate_creative_ideas():
    print("Testing generate_creative_ideas function.")
    sample_prompt = "Once upon a time,"
    results = generate_creative_ideas(sample_prompt, max_length=50, num_return_sequences=3)
    assert len(results) == 3, "Number of generated sequences should be 3."
    assert all('generated_text' in idea for idea in results), "Each entry in results should contain a 'generated_text' key."
    print("All test cases passed")