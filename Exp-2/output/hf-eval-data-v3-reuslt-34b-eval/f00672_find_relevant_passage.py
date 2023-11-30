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
    
    # Initialize a BERT model and a tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
    except OSError:
        return "OSError: There was a problem with loading the pre-trained model or tokenizer."
    
    # Split the input into a question and candidate passages using the given delimiters.
    
    try:
        question, *candidate_passages = [s.strip() for s in question.split("###")]
    except ValueError as e:
        return f"ValueError: {e} Please make sure you separate the question and candidate passages using '###'."
    
    # Find the most relevant passage by scoring each one and returning the highest-scored one.
    
    best_score, best_passage = 0.0, ""
    
    for passage in candidate_passages:
        
        input_text = f"[CLS] {question} [SEP] {passage} [SEP]"
        encoding = tokenizer(input_text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(model.device)
        
        with torch.no_grad():
            outputs = model(**encoding, return_dict=True)
            score = outputs.logits[0][0].item()
            
        if score > best_score:
            best_score = score
            best_passage = passage
    
    # Return the best-scoring passage.
    
    return best_passage

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