from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

def get_building_plan_info(question: str, building_plan_data: str) -> str:
    """
    This function uses a pretrained model from Hugging Face Transformers to answer questions based on a given building plan.
    
    Parameters:
    question (str): The question to be answered.
    building_plan_data (str): The building plan data.
    
    Returns:
    str: The answer to the question.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    # Prepare the inputs for the model
    inputs = tokenizer(question, building_plan_data, return_tensors='pt')
    result = model(**inputs)

    # Extract the answer from the model's output
    answer_start, answer_end = result.start_logits.argmax(), result.end_logits.argmax()
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end+1])

    return answer