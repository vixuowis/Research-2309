# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def find_relevant_passage(question: str, candidate_passages: list) -> str:
    """
    Find the most relevant passage given a question and several candidate passages.

    Args:
        question (str): The question to be answered.
        candidate_passages (list): A list of candidate passages.

    Returns:
        str: The most relevant passage.

    Raises:
        OSError: If there is a problem with loading the pre-trained model or tokenizer.
    """
    # load pretrained model and tokenizer for BERT-base
    try:
        model_name = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
    except OSError as e: print(e)

    # prepare the input data for BERT model (question + passages)
    question_inputs = tokenizer(f"question: {question} ", padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    passage_inputs = [tokenizer(" ".join([x["title"], x["text"]]), padding="max_length", truncation=True, max_length=512) for x in candidate_passages]

    # prepare the labels for BERT model (0: not relevant, 1: relevant)
    labels = torch.tensor(range(len(candidate_passages)))

    # calculate the prediction probabilities for each passage based on the question using BERT-base
    logits = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(candidate_passages), 8):
            batch = [question_inputs] + [x["input_ids"] for x in passage_inputs[i:i+4]]
            input_dict = {k: torch.stack([v[j] for v in batch], dim=0) for j, k in enumerate(batch[0].keys())}
            logits += model(**input_dict).logits.tolist()
    probabilities = [torch.sigmoid(torch.tensor([x])).item() for x in logits]

    # choose the most relevant passage based on its prediction probability by the BERT-base model (relevant > not relevant)
    return candidate_passages[probabilities.index(max(probabilities))]["title"]

# test_function_code --------------------

def test_find_relevant_passage():
    """
    Test the function find_relevant_passage.
    """
    question = 'How many people live in Berlin?'
    candidate_passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    assert isinstance(find_relevant_passage(question, candidate_passages), str)
    question = 'What is the capital of Germany?'
    candidate_passages = ['Berlin is the capital of Germany.', 'Paris is the capital of France.']
    assert find_relevant_passage(question, candidate_passages) == 'Berlin is the capital of Germany.'
    question = 'Who won the world cup in 2014?'
    candidate_passages = ['Germany won the world cup in 2014.', 'Brazil hosted the world cup in 2014.']
    assert find_relevant_passage(question, candidate_passages) == 'Germany won the world cup in 2014.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_find_relevant_passage()