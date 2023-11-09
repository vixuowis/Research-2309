import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

def generate_response_in_russian(input_text):
    '''
    This function generates a response in Russian to a given input text.
    It uses the pre-trained model 'tinkoff-ai/ruDialoGPT-medium' from Hugging Face Transformers.
    
    Parameters:
    input_text (str): The input text in Russian to which the function will generate a response.
    
    Returns:
    list: A list of generated responses.
    '''
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    # Load the pre-trained model
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')
    # Generate responses
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=3,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    # Decode the generated responses
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    return context_with_response