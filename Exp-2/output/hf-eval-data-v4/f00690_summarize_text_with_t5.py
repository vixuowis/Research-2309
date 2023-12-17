# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def summarize_text_with_t5(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5Model.from_pretrained('t5-base')
    input_text = 'summarize: ' + text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer('summarize:', return_tensors='pt').input_ids
    outputs = model.generate(input_ids, decoder_input_ids=decoder_input_ids)
    conclusion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return conclusion

# test_function_code --------------------

def test_summarize_text_with_t5():
    print('Testing started.')
    sample_text = 'Studies have been shown that owning a dog is good for you. Having a dog can help decrease stress levels, improve your mood, and increase physical activity.'

    print('Testing case [1/1] started.')
    summary = summarize_text_with_t5(sample_text)
    assert type(summary) == str, f"Test case [1/1] failed: Expected string output, got {type(summary)}"
    print('Summary:', summary)
    print('Testing finished.')