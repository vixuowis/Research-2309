# requirements_file --------------------

!pip install -U transformers optimum.onnxruntime

# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_story_en_to_fr(english_story: str) -> str:
    """
    Translate an English story into French using the 'optimum/t5-small' model.

    Args:
        english_story: A string containing the English story to be translated.

    Returns:
        A string containing the translated story in French.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    result = translator(english_story)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_story_en_to_fr():
    print("Testing started.")
    # Test case 1: Check if an empty string is handled correctly
    print("Testing case [1/3] started.")
    empty_story = ""
    empty_translation = translate_story_en_to_fr(empty_story)
    assert empty_translation == "", f"Test case [1/3] failed: Expected empty string, got {empty_translation}"

    # Test case 2: Translate a simple sentence
    print("Testing case [2/3] started.")
    simple_story = "The hero fights for justice."
    simple_translation = translate_story_en_to_fr(simple_story)
    assert isinstance(simple_translation, str), f"Test case [2/3] failed: Expected string, got {type(simple_translation)}"

    # Test case 3: Translate a complex sentence
    print("Testing case [3/3] started.")
    complex_story = "The enigmatic superhero, cloaked in shadow, silently defeats the villain."
    complex_translation = translate_story_en_to_fr(complex_story)
    assert isinstance(complex_translation, str), f"Test case [3/3] failed: Expected string, got {type(complex_translation)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_story_en_to_fr()