# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_language(text):
    # Load the language detection model using Transformers pipeline
    language_detector = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
    # Detect language
    result = language_detector(text)
    # Return the detected language and the confidence score
    return result[0]['label'], result[0]['score']

# test_function_code --------------------

def test_detect_language():
    # Testing the language detection function
    print("Testing language detection function.")

    # Sample texts in different languages
    sample_texts = ['Hello, how are you?', 'Hola, cómo estás?', 'Hallo, wie geht es dir?', 'Bonjour, comment ça va?']

    # Expected results based on the sample texts
    expected_languages = ['en', 'es', 'de', 'fr']

    # Perform detection and assertion
    for i, text in enumerate(sample_texts):
        detected_language, _ = detect_language(text)
        assert detected_language == expected_languages[i], f"Test case [" + str(i + 1) + "/4] failed: Detected language {{detected_language}} does not match expected language {{expected_languages[i]}}."

    print("All test cases passed for language detection function.")

test_detect_language()