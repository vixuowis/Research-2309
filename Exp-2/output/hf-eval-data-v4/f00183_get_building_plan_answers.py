# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_building_plan_answers(question, building_plan_data):
    """
    Answer a specific question about a building plan using a pre-trained document question answering model.

    :param question: str, the question to be answered
    :param building_plan_data: str, the context data from the building plan
    :return: str, the extracted answer
    """
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    inputs = tokenizer(question, building_plan_data, return_tensors='pt')
    result = model(**inputs)
    answer_start, answer_end = result.start_logits.argmax(), result.end_logits.argmax()
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end + 1])
    return answer

# test_function_code --------------------

def test_get_building_plan_answers():
    print("Testing get_building_plan_answers function.")

    # Assuming 'building_plan_sample' is a string containing building plan data for testing
    building_plan_sample = 'Building plan data here...'

    # Test case: Retrieve the number of floors in the building
    question1 = 'How many floors does the building have?'
    answer1 = get_building_plan_answers(question1, building_plan_sample)
    assert answer1.isdigit(), f'Test case failed: Expected a numeric answer, got {answer1}'

    # Test case: Retrieve the total estimated cost of the project
    question2 = 'What is the total estimated cost of the project?'
    answer2 = get_building_plan_answers(question2, building_plan_sample)
    assert answer2.startswith('$'), f'Test case failed: Expected a cost answer starting with $, got {answer2}'

    print('All tests passed!')

# Run the test function
test_get_building_plan_answers()