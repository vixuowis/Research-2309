# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_game_day(context: str, question: str) -> str:
    """
    This function uses the Hugging Face Transformers pipeline for question answering to extract the day of the game from the given context.

    Args:
        context (str): The context in which the game was played.
        question (str): The question to be answered.

    Returns:
        str: The day on which the game was played.
    """
    
    # Load Hugging Face pipeline and perform inference
    game_day = pipeline(task="question-answering")({"context": context, "question": question})["answer"]
    
    return game_day

# test_function_code --------------------

def test_get_game_day():
    assert get_game_day("The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.", "What day was the game played on?") == 'February 7, 2016'
    assert get_game_day("The match took place on March 3, 2020 at the National Stadium.", "When was the match?") == 'March 3, 2020'
    assert get_game_day("The event occurred on December 25, 2019 at the Madison Square Garden.", "When did the event occur?") == 'December 25, 2019'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_game_day()