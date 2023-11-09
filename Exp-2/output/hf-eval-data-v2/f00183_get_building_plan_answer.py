# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_building_plan_answer(question: str, building_plan_data: str) -> str:
    """
    This function uses a pretrained model from Hugging Face Transformers to answer questions based on a given building plan.

    Args:
        question (str): The question to be answered.
        building_plan_data (str): The building plan data to extract the answer from.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    inputs = tokenizer(question, building_plan_data, return_tensors='pt')
    result = model(**inputs)
    answer_start, answer_end = result.start_logits.argmax(), result.end_logits.argmax()

    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end+1])
    return answer

# test_function_code --------------------

def test_get_building_plan_answer():
    """
    This function tests the 'get_building_plan_answer' function with a sample question and building plan data.
    """
    question = 'What is the total estimated cost of the project?'
    building_plan_data = 'Building plan data here...'
    answer = get_building_plan_answer(question, building_plan_data)
    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_get_building_plan_answer()