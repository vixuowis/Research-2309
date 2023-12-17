# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def answer_questions_from_building_plan(question, building_plan_data):
    """
    Answers the questions provided using a LayoutLMX model pretrained on document QA tasks,
    based on the context given through building plan data.

    Args:
        question (str): The question to be answered.
        building_plan_data (str): The document containing the context for answering the question.

    Returns:
        str: The extracted answer from the document.

    Raises:
        ValueError: If the inputs are not strings or if they are empty.
    """
    if not question or not building_plan_data:
        raise ValueError("Question and building plan data must be provided.")

    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    inputs = tokenizer(question, building_plan_data, return_tensors='pt')
    result = model(**inputs)
    answer_start, answer_end = result.start_logits.argmax(), result.end_logits.argmax()

    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end+1])

    return answer

# test_function_code --------------------

def test_answer_questions_from_building_plan():
    print("Testing started.")
    # Since this process leverages a pretrained model and does not generate a dataset by itself, we'll simulate a case
    sample_question = "How many floors does the building have?"
    sample_building_plan_data = "The building plan has 5 floors including a rooftop garden."

    # Testing case 1
    print("Testing case [1/1] started.")
    assert answer_questions_from_building_plan(sample_question, sample_building_plan_data) == "5", \
           "Test case [1/1] failed: Expected '5', got " + answer_questions_from_building_plan(sample_question, sample_building_plan_data)
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_questions_from_building_plan()