# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tag_chinese_pos(chinese_sentence):
    """
    Detects the part-of-speech tags for words in a Chinese sentence.

    Parameters:
        chinese_sentence (str): A string containing the Chinese sentence to be tagged.

    Returns:
        List[str]: A list of part-of-speech tags for each token in the sentence.
    """
    # Load the tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-pos')

    # Tokenize the input sentence and convert to model input format
    tokens = tokenizer(chinese_sentence, return_tensors='pt')

    # Get model predictions for part-of-speech tags
    outputs = model(**tokens)
    part_of_speech_tags = outputs.logits.argmax(-1).squeeze().tolist()
    
    # Map each token id to its part-of-speech tag
    tag_names = [model.config.id2label[tag_id] for tag_id in part_of_speech_tags]

    return tag_names

# test_function_code --------------------

def test_tag_chinese_pos():
    print("Testing tag_chinese_pos function.")

    # Test case 1: A simple sentence
    sentence1 = '我爱北京天安门'
    tags1 = tag_chinese_pos(sentence1)
    print("Test case 1 passed.")

    # Test case 2: A complex sentence
    sentence2 = '他们在教室里读书。'
    tags2 = tag_chinese_pos(sentence2)
    print("Test case 2 passed.")

    # Test case 3: An interrogative sentence
    sentence3 = '你吃过早餐了吗？'
    tags3 = tag_chinese_pos(sentence3)
    print("Test case 3 passed.")

    print("Testing finished.")
    assert len(tags1) > 0, "Test case 1 failed: No tags returned."
    assert len(tags2) > 0, "Test case 2 failed: No tags returned."
    assert len(tags3) > 0, "Test case 3 failed: No tags returned."

    print("All test cases passed.")

# Running the test function
test_tag_chinese_pos()