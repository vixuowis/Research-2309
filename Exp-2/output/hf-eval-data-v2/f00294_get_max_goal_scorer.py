# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def get_max_goal_scorer(question: str, table: str) -> str:
    """
    This function identifies the player who has scored the maximum goals in a given match.

    Args:
        question (str): The question to be answered. For example, "What player scored the most goals?"
        table (str): The table data in CSV format. For example, "Player,Goals\nA,2\nB,3\nC,1"

    Returns:
        str: The name of the player who scored the most goals.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-sqa')
    inputs = tokenizer(question, table, return_tensors="pt")
    outputs = model(**inputs)
    answer_label = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(axis=2)[0, 0])
    return answer_label

# test_function_code --------------------

def test_get_max_goal_scorer():
    """
    This function tests the get_max_goal_scorer function.
    """
    question = "What player scored the most goals?"
    table = "Player,Goals\nA,2\nB,3\nC,1"
    assert get_max_goal_scorer(question, table) == 'B'

# call_test_function_code --------------------

test_get_max_goal_scorer()