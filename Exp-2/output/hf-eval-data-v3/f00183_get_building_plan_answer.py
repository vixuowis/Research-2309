# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_building_plan_answer(question: str, building_plan_data: str) -> str:
    """
    This function uses a pretrained model from Hugging Face to answer questions based on a given building plan.

    Args:
        question (str): The question to be answered.
        building_plan_data (str): The building plan data.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If the model or tokenizer cannot be loaded from the Hugging Face repository.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
        model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    except Exception as e:
        raise OSError('Model or tokenizer cannot be loaded from the Hugging Face repository.') from e

    inputs = tokenizer(question, building_plan_data, return_tensors='pt')
    result = model(**inputs)
    answer_start, answer_end = result.start_logits.argmax(), result.end_logits.argmax()

    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end+1])
    return answer

# test_function_code --------------------

def test_get_building_plan_answer():
    """
    This function tests the get_building_plan_answer function.
    """
    question = 'What is the total estimated cost of the project?'
    building_plan_data = 'The total estimated cost of the project is $1,000,000.'
    answer = get_building_plan_answer(question, building_plan_data)
    assert answer == '$1,000,000', f'Error: {answer}'

    question = 'How many floors does the building have?'
    building_plan_data = 'The building has 5 floors.'
    answer = get_building_plan_answer(question, building_plan_data)
    assert answer == '5', f'Error: {answer}'

    question = 'What is the area size of the building?'
    building_plan_data = 'The area size of the building is 5000 sq ft.'
    answer = get_building_plan_answer(question, building_plan_data)
    assert answer == '5000 sq ft', f'Error: {answer}'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_building_plan_answer()