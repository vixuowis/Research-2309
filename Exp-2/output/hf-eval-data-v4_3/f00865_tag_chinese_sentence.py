# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tag_chinese_sentence(chinese_sentence: str) -> list:
    """
    Tags each word in a Chinese sentence with its corresponding part of speech.

    Args:
        chinese_sentence (str): A string containing a Chinese sentence to be tagged.

    Returns:
        list: A list containing tuples with the word and its part of speech tag.

    Raises:
        ValueError: If the sentence is empty.

    """
    if not chinese_sentence:
        raise ValueError('The sentence cannot be empty.')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-pos')
    tokens = tokenizer(chinese_sentence, return_tensors='pt')
    outputs = model(**tokens)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tags = [model.config.id2label[p] for p in predictions]
    tagged_sentence = list(zip(tokens.tokens()[1:-1], tags[1:-1]))
    return tagged_sentence

# test_function_code --------------------

def test_tag_chinese_sentence():
    print("Testing started.")
    # Test case 1: Non-empty sentence
    print("Testing case [1/3] started.")
    sentence = '我爱北京天安门'
    tagged_sentence = tag_chinese_sentence(sentence)
    assert tagged_sentence is not None, f"Test case [1/3] failed: Expected non-empty result, got {tagged_sentence}"
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in tagged_sentence), f"Test case [1/3] failed: Expected list of tuples with word and tag, got {tagged_sentence}"

    # Test case 2: Empty input
    print("Testing case [2/3] started.")
    try:
        tag_chinese_sentence('')
        assert False, "Test case [2/3] failed: Expected ValueError for empty input."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_tag_chinese_sentence()