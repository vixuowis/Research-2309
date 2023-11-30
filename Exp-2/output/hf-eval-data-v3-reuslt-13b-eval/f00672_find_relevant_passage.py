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
    
    # Initialize our model and tokenizer
    try:
        
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", return_tensors="pt")
        model = AutoModelForSequenceClassification.from_pretrained("deepset/roberta-base-squad2")
        
    except OSError as ose:
    
        print(f"Problem with loading the pre-trained model or tokenizer: {ose}")
        return ""
            
    # Tokenize our inputs and format them for PyTorch
    question = [question] * len(candidate_passages)
    inputs = tokenizer.encode_plus(question, candidate_passages, add_special_tokens=True, max_length=512, return_tensors='pt')
    
    # Get our input ids and attention masks
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Set the batch size to 1 so we can add more samples if needed (relevant for production)
    batch_size = 1
    
    # Wrap our tensors in a PyTorch DataLoader
    data = TensorDataset(input_ids, attention_mask)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    # Run through our model and get the predicted start/end token for the answer
    all_results = []
    for batch in tqdm(dataloader):
        
        input_ids, attention_mask = batch
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                start_logits, end_logits = outputs[0], outputs[1]
                    
                # Convert the predicted token indices into text
                tokens = input_ids[0].cpu().numpy()
                
                # Decode the output
                all_tokens = tokenizer.convert_ids_to_tokens(tokens)

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