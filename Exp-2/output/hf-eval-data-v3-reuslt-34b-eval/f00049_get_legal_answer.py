# function_import --------------------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# function_code --------------------

def get_legal_answer(question: str, context: str) -> str:
    """
    This function uses a pretrained model from Hugging Face Transformers to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.

    Returns:
        str: The answer to the question based on the context.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    
    input_ids = tokenizer(question, context, return_tensors="pt").input_ids
    start_scores, end_scores = model(input_ids)[:2]
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    
    if answer_end >= len(input_ids[0]):
        return ""
    else:
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[answer_start:answer_end]))

# test_function_code --------------------

def test_get_legal_answer():
    question = 'Who is the licensee?'
    context = 'We hereby grant the Licensee the exclusive right to develop, construct, operate and promote the Project, as well as to manage the daily operations of the Licensed Facilities during the Term. In consideration for the grant of the License, the Licensee shall pay to the Licensor the full amount of Ten Million (10,000,000) Dollars within thirty (30) days after the execution hereof.'
    answer = get_legal_answer(question, context)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert answer != '', 'The answer should not be an empty string.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_legal_answer()