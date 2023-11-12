# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_response(instruction: str, knowledge: str, dialog: list) -> str:
    '''
    Generate a response based on the given instruction, knowledge and dialog.

    Args:
        instruction (str): The instruction for the response.
        knowledge (str): The knowledge to be used in generating the response.
        dialog (list): The dialog context for the response.

    Returns:
        str: The generated response.
    '''
    model_name = 'microsoft/GODEL-v1_1-base-seq2seq'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog} {knowledge}'
    input_ids = tokenizer(f'{query}', return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# test_function_code --------------------

def test_generate_response():
    '''
    Test the generate_response function.
    '''
    instruction = 'what is the best way to choose a video game?'
    knowledge = 'Some factors to consider when choosing a video game are personal preferences, genre, graphics, gameplay, storyline, platform, and reviews from other players or gaming websites.'
    dialog = ['What type of video games do you prefer playing?', 'I enjoy action-adventure games and a decent storyline.']
    response = generate_response(instruction, knowledge, dialog)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_response()