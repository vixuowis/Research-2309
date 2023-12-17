# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import TextGenerationPipeline, Bloom7b1Model

# function_code --------------------

def generate_houseplant_care_tips(prompt):
    """
    Generate a paragraph with tips on how to take care of houseplants using a pre-trained language model.
    
    Args:
        prompt (str): A prompt text to guide the model in generating houseplant care tips.
    
    Returns:
        str: A generated paragraph with houseplant care tips.
    """
    model = Bloom7b1Model.from_pretrained('bigscience/bloom-7b1')
    text_generator = TextGenerationPipeline(model=model)
    generated_paragraph = text_generator(prompt, max_length=150, do_sample=True)[0]['generated_text']
    return generated_paragraph

# test_function_code --------------------

def test_generate_houseplant_care_tips():
    print("Testing started.")

    # Test case 1: Check if the function generates a string
    prompt = "Tips on how to take care of houseplants:"
    generated_text = generate_houseplant_care_tips(prompt)
    assert isinstance(generated_text, str), f"Test case failed: The output is not a string"

    # Test case 2: Check if the generated text is not empty
    assert len(generated_text) > 0, f"Test case failed: The generated text is empty"

    # Test case 3: Check if the prompt is in the generated text
    assert prompt in generated_text, f"Test case failed: The prompt is not included in the generated text"
    print("Testing finished.")