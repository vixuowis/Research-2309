import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

def generate_dialogue():
    '''
    This function generates a dialogue in Russian using the pretrained model 'tinkoff-ai/ruDialoGPT-medium'.
    The dialogue covers a general greeting and asking about the users' well-being.
    '''
    # Load the tokenizer and the pretrained model
    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

    # Prepare the input text
    inputs = tokenizer('@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела?', return_tensors='pt')

    # Generate the dialogue
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

    # Decode the generated token ids back to text
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]

    return context_with_response