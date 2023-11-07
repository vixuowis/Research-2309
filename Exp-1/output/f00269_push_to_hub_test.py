from f00269_push_to_hub import *
def test_push_to_hub():
    # Initialize tokenizer
    tokenizer = PreTrainedTokenizer()

    # Call push_to_hub function
    callback = push_to_hub('my_awesome_qa_model', tokenizer)

    # Assert statements
    assert isinstance(callback, PushToHubCallback)
    assert callback.output_dir == 'my_awesome_qa_model'
    assert callback.tokenizer == tokenizer


# Test the function
if __name__ == '__main__':
    test_push_to_hub()
