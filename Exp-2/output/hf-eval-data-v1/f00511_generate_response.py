from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

def generate_response(instruction, knowledge, dialog):
    """
    This function generates a response for the given instruction, knowledge, and dialog using the GODEL model.
    
    Parameters:
    instruction (str): The instruction for the model.
    knowledge (str): The knowledge for the model.
    dialog (list): The dialog history.
    
    Returns:
    str: The generated response.
    """
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog} {knowledge}'
    input_ids = tokenizer(f'{query}', return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output