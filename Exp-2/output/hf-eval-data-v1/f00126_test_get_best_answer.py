def test_get_best_answer():
    # Define a test question and a list of possible answers
    question = 'How many people live in Berlin?'
    passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']

    # Call the function with the test data
    best_answer = get_best_answer(question, passages)

    # Check if the function returns the correct answer
    assert best_answer == 'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'Test failed: The function returned the wrong answer.'

    print('Test passed: The function returned the correct answer.')

# Call the test function
test_get_best_answer()