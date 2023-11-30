# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def answer_question(question: str, context: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer a question based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is asked.

    Returns:
        str: The answer to the question.
    """
    # Set up model and tokenizer ----------------
    
    model_name = "mrm8488/spanbert-finetuned-squadv2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize ----------------------------
    
    inputs = tokenizer(question, context, add_special_tokens=True, truncation="only_second", return_tensors="pt")
    
    input_ids = inputs["input_ids"].tolist()[0]
    attention_mask = inputs["attention_mask"].tolist()[0]
    
    # Get answer -------------------------------
    
    outputs = model(**inputs)
    answer_start_scores=outputs.start_logits
    answer_end_scores=outputs.end_logits
    start_index=answer_start_scores.argmax() 
    end_index=answer_end_scores.argmax()
    
    token_ids=[tokenizer.convert_ids_to_tokens(i) for i in input_ids]
    answer=""
    if len(token_ids[0])>start_index:
        answer+=" ".join(token_ids[0][start_index:end_index + 1])
    
    # Return ------------------------------------
    
    return answer.strip()


# test_function_code --------------------

def test_answer_question():
    """
    This function tests the answer_question function with some test cases.
    """
    question1 = 'What is the capital of France?'
    context1 = 'Paris is the capital of France.'
    assert answer_question(question1, context1) == 'Paris'

    question2 = 'Who won the world cup in 2018?'
    context2 = 'The 2018 FIFA World Cup was won by France.'
    assert answer_question(question2, context2) == 'France'

    question3 = 'Who is the CEO of Tesla?'
    context3 = 'Elon Musk is the CEO of Tesla.'
    assert answer_question(question3, context3) == 'Elon Musk'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_answer_question()