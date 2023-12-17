# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review(review_text, categories):
    # Loads the zero-shot classification pipeline
    classifier = pipeline('zero-shot-classification', model='vicgalle/xlm-roberta-large-xnli-anli')
    # Performs classification
    result = classifier(review_text, categories)
    # The result is a dictionary containing labels and scores. We return the label with the highest score
    return max(result['labels'], key=lambda label: result['scores'][result['labels'].index(label)])

# test_function_code --------------------

def test_classify_review():
    print("Testing started.")
    # Test case 1: A review related to traveling.
    travel_text = "El viaje fue increíble, visité muchos lugares hermosos."
    assert classify_review(travel_text, ['viaje', 'cocina', 'danza']) == 'viaje', "Test case [1/3] failed: Expected 'viaje'"

    # Test case 2: A review related to cooking.
    cooking_text = "Me encanta preparar recetas nuevas y sabrosas."
    assert classify_review(cooking_text, ['viaje', 'cocina', 'danza']) == 'cocina', "Test case [2/3] failed: Expected 'cocina'"

    # Test case 3: A review related to dancing.
    dancing_text = "Anoche fuimos a bailar salsa toda la noche."
    assert classify_review(dancing_text, ['viaje', 'cocina', 'danza']) == 'danza', "Test case [3/3] failed: Expected 'danza'"
    print("Testing finished.")

# Testing the function
test_classify_review()