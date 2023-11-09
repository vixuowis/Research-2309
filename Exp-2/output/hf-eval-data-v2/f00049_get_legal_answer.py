# function_import --------------------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def get_legal_answer(question: str, context: str) -> str:
    """
    This function uses a pretrained model 'Rakib/roberta-base-on-cuad' from Hugging Face Transformers
    to answer questions based on a given context. The model is trained on the CUAD dataset for
    question answering tasks on legal documents.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question based on the context.
    """
    tokenizer = AutoTokenizer.from_pretrained('Rakib/roberta-base-on-cuad')
    model = AutoModelForQuestionAnswering.from_pretrained('Rakib/roberta-base-on-cuad')
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

# test_function_code --------------------

def test_get_legal_answer():
    """
    This function tests the 'get_legal_answer' function with a sample question and context.
    """
    question = 'What is the amount to be paid by the Licensee to the Licensor?'
    context = 'We hereby grant the Licensee the exclusive right to develop, construct, operate and promote the Project, as well as to manage the daily operations of the Licensed Facilities during the Term. In consideration for the grant of the License, the Licensee shall pay to the Licensor the full amount of Ten Million (10,000,000) Dollars within thirty (30) days after the execution hereof.'
    answer = get_legal_answer(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer == 'Ten Million (10,000,000) Dollars', 'The function returned the wrong answer.'

# call_test_function_code --------------------

test_get_legal_answer()