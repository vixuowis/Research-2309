def test_memory_game():
    '''
    This function tests the memory_game function using a sample from the SQuAD v1.1 dataset.
    '''
    context = 'Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.'
    question = 'What is a good example of a question answering dataset?'
    user_answer = 'SQuAD dataset'
    
    assert memory_game(context, question, user_answer) == 'Correct!', 'Test Case 1 Failed'
    
    context = 'DistilBERT base uncased distilled SQuAD is a fine-tuned version of DistilBERT-base-uncased for the task of question answering.'
    question = 'What is DistilBERT base uncased distilled SQuAD?'
    user_answer = 'A fine-tuned version of DistilBERT-base-uncased for the task of question answering'
    
    assert memory_game(context, question, user_answer) == 'Correct!', 'Test Case 2 Failed'
    
    print('All test cases pass')

# Run the test function
test_memory_game()