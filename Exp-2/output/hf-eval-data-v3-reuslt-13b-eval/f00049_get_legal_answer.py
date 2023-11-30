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

    # Get model and tokenizer from Hugging Face Hub
    model = AutoModelForQuestionAnswering.from_pretrained(
        "ktrapeznikov/albert-xlarge-v2-squad-v2")
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

    # Tokenize inputs
    encoding = tokenizer(question, context)
    input_ids = torch.tensor([[encoding.input_ids]])
    attention_mask = torch.tensor([[encoding.attention_mask]])

    # Get answers from model
    output = model(input_ids, attention_mask=attention_mask)
    answer_start_scores = output.start_logits
    answer_end_scores = output.end_logits

    # Get string from tokenized answer indices
    answer_tokens = encoding.tokens[torch.argmax(answer_start_scores) : torch.argmax(answer_end_scores)+1]
    return " ".join([t for t in answer_tokens if t != "[CLS]" and t != "[SEP]"])

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