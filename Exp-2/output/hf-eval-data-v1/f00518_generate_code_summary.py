from transformers import RobertaTokenizer, T5ForConditionalGeneration


def generate_code_summary(code_snippet):
    """
    This function generates a short summary of the provided code snippet using the Salesforce/codet5-base model.
    
    Parameters:
    code_snippet (str): The code snippet to summarize.
    
    Returns:
    str: The generated summary of the code snippet.
    """
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    input_ids = tokenizer(code_snippet, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=25)
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary