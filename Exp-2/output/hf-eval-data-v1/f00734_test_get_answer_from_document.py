def test_get_answer_from_document():
    """
    This function tests the 'get_answer_from_document' function.
    It uses a sample document and a set of questions to test the function.
    """
    # Sample document
    document = "Our hotel offers a variety of rooms. The cost of a deluxe suite per night is $200."
    
    # Set of questions
    questions = ["What is the cost of a deluxe suite per night?", "What rooms does the hotel offer?"]
    
    # Expected answers
    expected_answers = ["$200", "a variety of rooms"]
    
    for i, question in enumerate(questions):
        # Get the answer from the document
        answer = get_answer_from_document(question, document)
        
        # Check if the answer is correct
        assert answer == expected_answers[i], f"Expected '{expected_answers[i]}', but got '{answer}'"

test_get_answer_from_document()