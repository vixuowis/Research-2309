# requirements_file --------------------

!pip install -U transformers summarization

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def create_quiz_from_article(article_text):
    '''
    Summarize an article text into a paragraph, generate a question from the summary and provide multiple choices.
    Then determine the correct choice using a question-answering model.

    Parameters:
    article_text (str): The article text to be summarized and used to generate the quiz.

    Returns:
    tuple: A tuple with the summary paragraph, the generated question, choices, and the correct answer.
    '''
    # Use a summarization model to create the summary
    summary_pipeline = pipeline('summarization')
    summary_text = summary_pipeline(article_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    # Placeholder for generating questions and options
    question = 'What is the main topic of the summarized text?'
    options = ['Science', 'Technology', 'Economics', 'Literature']

    # Instantiate the Question Answering pipeline
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')

    # Determine the correct choice
    predictions = []
    for option in options:
        input_data = {'context': summary_text, 'question': question + ' ' + option}
        result = qa_pipeline(input_data)
        predictions.append((option, result['score']))

    correct_answer = max(predictions, key=lambda x: x[1])[0]

    return (summary_text, question, options, correct_answer)

# test_function_code --------------------

def test_create_quiz_from_article():
    print("Testing started.")
    # A sample article text
    sample_article = "The recent advances in machine learning and artificial intelligence ..."

    # Testing the quiz creation
    summary, question, options, correct_answer = create_quiz_from_article(sample_article)

    # Test case: Check the summary is not empty
    assert summary, "Test case failed: The summary is empty."

    # Test case: Check that four options are generated
    assert len(options) == 4, "Test case failed: There should be four options."

    # Test case: Check the correct_answer is in options
    assert correct_answer in options, "Test case failed: The correct answer is not in the options."

    print("Testing finished.")

# Run the test
if __name__ == '__main__':
    test_create_quiz_from_article()