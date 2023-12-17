# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spanish_article(article_text):
    # Load the zero-shot classification pipeline
    classifier = pipeline('zero-shot-classification', model='Recognai/bert-base-spanish-wwm-cased-xnli')
    # Define the candidate labels for classification
    candidate_labels = ['cultura', 'sociedad', 'economia', 'salud', 'deportes']
    # Define the hypothesis template for zero-shot classification
    hypothesis_template = 'Este texto trata de {}.'
    # Classify the article
    predictions = classifier(article_text, candidate_labels, hypothesis_template=hypothesis_template)
    # Return the predictions
    return predictions

# test_function_code --------------------

def test_classify_spanish_article():
    print('Testing started.')
    sample_data = 'El autor se perfila, a los 50 a\xf1os de su muerte, como uno de los grandes de su siglo'
    # Call the function with a sample article
    prediction = classify_spanish_article(sample_data)
    # Retrieve the top prediction label
    top_prediction = prediction['labels'][0]
    assert top_prediction in ['cultura', 'sociedad', 'economia', 'salud', 'deportes'], f'Test failed: Unexpected label {top_prediction}'
    print('Testing finished.')

# Run the test function
test_classify_spanish_article()