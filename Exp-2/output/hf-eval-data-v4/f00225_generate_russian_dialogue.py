# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_russian_dialogue(greeting: str, follow_up: str) -> list:
    """
    Generate a dialogue in Russian where the first participant greets and the second participant asks about well-being.

    :param greeting: Greeting text by the first participant.
    :param follow_up: Follow-up question about well-being by the second participant.
    :return: A list of generated dialogue responses.
    """
    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

    # Prepare the conversation
    conversation = f'@@ПЕРВЫЙ@@ {greeting} @@ВТОРОЙ@@ {follow_up} @@ПЕРВЫЙ@@'
    inputs = tokenizer(conversation, return_tensors='pt')

    # Generate dialogue
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
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=40
    )

    # Decode the generated dialogue
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    return context_with_response

# test_function_code --------------------

def test_generate_russian_dialogue():
    print('Testing started.')

    # Test case 1: Check if the function returns three dialogues
    print('Testing case [1/1] started.')
    greetings = 'привет'
    follow_up = 'как дела?'
    responses = generate_russian_dialogue(greetings, follow_up)
    assert len(responses) == 3, f'Test case [1/1] failed: Expected 3 dialogues, got {len(responses)}'
    print('All test cases passed.')

# Run the test function
test_generate_russian_dialogue()