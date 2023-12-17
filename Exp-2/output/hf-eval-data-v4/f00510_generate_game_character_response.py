# requirements_file --------------------

!pip install -U torch, transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# function_code --------------------

def generate_game_character_response(user_input, chat_history_ids=None, step=0):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelWithLMHead.from_pretrained('output-small')
    user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if step > 0 else user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
    ai_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return ai_response, chat_history_ids

# test_function_code --------------------

def test_generate_game_character_response():
    print("Testing started.")
    sample_inputs = ["Hello!", "How are you?", "What should we do next?"]

    chat_history_ids = None
    for step, user_input in enumerate(sample_inputs):
        print(f"Testing case [{step+1}/{len(sample_inputs)}] started.")
        response, chat_history_ids = generate_game_character_response(user_input, chat_history_ids, step)
        assert response, f"Test case [{step+1}/{len(sample_inputs)}] failed: No response returned."
    print("Testing finished.")

# Run the test function
test_generate_game_character_response()