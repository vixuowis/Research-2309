# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelWithLMHead, AutoTokenizer

# function_code --------------------

def gen_sentence(words, max_length=32):
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    input_text = ' '.join(words)
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_gen_sentence():
    print("Testing started.")
    test_cases = [
        (["tree", "plant", "ground", "hole", "dig"], "Dig a hole in the ground to plant a tree"),
        (["sun", "sky", "dawn"], "The sun rises at dawn, lighting up the sky"),
        (["book", "read", "knowledge"], "Reading books expands knowledge")
    ]

    for i, (words, expected) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/3] started.")
        result = gen_sentence(words)
        assert result == expected, f"Test case [{i}/3] failed: Expected '{{expected}}' but got '{{result}}'"
        print(f"Testing case [{i}/3] succeeded.")

    print("Testing finished.")

test_gen_sentence()