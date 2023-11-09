from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

def generate_chatbot_response(instruction: str, knowledge: str, dialog: list) -> str:
    """
    Generate a response from the chatbot based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): The user's input.
        knowledge (str): Relevant external information.
        dialog (list): The previous dialog context.

    Returns:
        str: The generated output from the chatbot.
    """
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output