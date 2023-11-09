# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_dialogue(input_text):
    """
    This function generates a dialogue in Russian using the pretrained model 'tinkoff-ai/ruDialoGPT-medium'.

    Args:
        input_text (str): The input text in Russian to start the conversation.

    Returns:
        list: A list of generated dialogues.
    """
    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    inputs = tokenizer(input_text, return_tensors='pt')
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
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    return context_with_response

# test_function_code --------------------

def test_generate_dialogue():
    """
    This function tests the 'generate_dialogue' function by providing a sample input and checking the output type.
    """
    input_text = '@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела?'
    output = generate_dialogue(input_text)
    assert isinstance(output, list), 'Output should be a list.'
    assert all(isinstance(i, str) for i in output), 'All elements in the output list should be strings.'

# call_test_function_code --------------------

test_generate_dialogue()