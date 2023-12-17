# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelWithLMHead, AutoTokenizer

# function_code --------------------

def generate_creative_sentence(words, max_length=32):
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_creative_sentence():
    print("Testing started.")
    words = "moon rabbit forest magic"
    result = generate_creative_sentence(words)
    assert isinstance(result, str), f"Test case failed: The result is not a string."
    assert all(word in result for word in words.split()), f"Test case failed: Not all input words are used in the generated sentence."
    print("Test case passed: Generated sentence is a string containing all input words.")
    print("Testing finished.")

# Run the test function
test_generate_creative_sentence()