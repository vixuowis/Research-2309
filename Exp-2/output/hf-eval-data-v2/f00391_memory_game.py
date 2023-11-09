# function_import --------------------

from transformers import pipeline

# function_code --------------------

def memory_game(context: str, question: str, user_answer: str) -> str:
    '''
    This function implements a memory game using a question answering model.
    
    Args:
        context (str): The hidden context that the player needs to remember.
        question (str): The question you want to ask the player based on the context.
        user_answer (str): The answer provided by the user.
    
    Returns:
        str: A message indicating whether the user's answer is correct or not.
    '''
    # Load the model
    question_answerer = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    
    # Check the correctness of the answer
    result = question_answerer(question=question, context=context)
    predicted_answer = result['answer']
    
    if user_answer.lower() == predicted_answer.lower():
        return 'Correct!'
    else:
        return f'Incorrect. The correct answer is: {predicted_answer}'

# test_function_code --------------------

def test_memory_game():
    '''
    This function tests the memory_game function.
    '''
    context = 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.'
    question = 'What is the Eiffel Tower made of?'
    user_answer = 'wrought-iron'
    assert 'Correct' in memory_game(context, question, user_answer)
    
    user_answer = 'steel'
    assert 'Incorrect' in memory_game(context, question, user_answer)

# call_test_function_code --------------------

test_memory_game()