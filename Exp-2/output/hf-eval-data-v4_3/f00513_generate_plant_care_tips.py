# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TextGenerationPipeline, Bloom7b1Model

# function_code --------------------

def generate_plant_care_tips(prompt: str) -> str:
    '''
    Generates a paragraph with tips on taking care of houseplants using a pretrained language model.

    Args:
        prompt (str): The input text prompt to generate tips from.

    Returns:
        str: Generated paragraph with houseplant care tips.

    Raises:
        Exception: If the model fails to generate the text.
    '''
    try:
        model = Bloom7b1Model.from_pretrained('bigscience/bloom-7b1')
        text_generator = TextGenerationPipeline(model=model)
        generated_paragraph = text_generator(prompt)[0]['generated_text']
        return generated_paragraph
    except Exception as e:
        raise Exception(f'Error while generating text: {str(e)}')

# test_function_code --------------------

def test_generate_plant_care_tips():
    print("Testing started.")
    prompt = "Tips on how to take care of houseplants:"

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    generated_tips = generate_plant_care_tips(prompt)
    assert isinstance(generated_tips, str), f"Test case [1/1] failed: Output is not a string."
    assert len(generated_tips) > 0, f"Test case [1/1] failed: No content generated."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_plant_care_tips()