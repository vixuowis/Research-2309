# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_response(instruction: str, knowledge: str, dialog: list) -> str:
    """
    Generate a response based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): Instruction on how to respond.
        knowledge (str): Knowledge about the situation.
        dialog (list): List of dialogues.

    Returns:
        str: Generated response.
    """
    
    # Create a tokenizer for summarization, and a model for it.
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-xsum-12-6')
    
    # Tokenize input text and generate IDs using the tokenizer.
    ids_tokens = []
    ids_tokens.append(tokenizer.encode("{}: {}".format('Knowledge', knowledge), add_special_tokens=False))
    for i in range(len(dialog)):
        # print(i)
        if i < len(dialog)-1:
            text = "{}{}".format(dialog[i], dialog[i+1])
        else: 
            text = "{}".format(dialog[i])
            
        ids_tokens.append(tokenizer.encode("{}: {}".format('User', text), add_special_tokens=False))
    
    # Generate a response based on the instruction and dialogue.
    response = ""
    for i in range(len(ids_tokens)):
        if i == 0:
            ids = tokenizer("summarize: {}".format(' '.join(ids_tokens[i])), return_tensors='pt', truncation=True, max_length=1024)['input_ids']
            
        else:
            ids = tokenizer("summarize: {}".format(' '.join(ids_tokens[i])), return_tensors='pt')['input_ids']
        
        # Generate response.
        outs = model.generate(ids, max_length=1024)
        resp = [tokenizer.decode(ids) for ids in outs]
        
        if i == 0:
            response += resp[0].replace('summarize: ', '').replace("'","’")
            
        else:
            response += '\n' + resp[0].replace('summarize: ', '').replace("'

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    instruction = 'How can I respond to a customer complaint about late delivery?'
    knowledge = 'The courier had external delays due to bad winter weather.'
    dialog = ['Customer: My package is late. What is going on?', 'Support: I apologize for the inconvenience. I will check what is happening with the package and get back to you.']
    response = generate_response(instruction, knowledge, dialog)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()