from f00301_push_to_hub_callback import *
def test_push_to_hub_callback():
    # Test case 1
    output_dir = "my_awesome_eli5_clm-model"
    tokenizer = "tokenizer"
    callback = push_to_hub_callback(output_dir, tokenizer)
    assert isinstance(callback, PushToHubCallback)

    # Test case 2
    output_dir = "another_model"
    tokenizer = "another_tokenizer"
    callback = push_to_hub_callback(output_dir, tokenizer)
    assert isinstance(callback, PushToHubCallback)

    # Test case 3
    output_dir = "model3"
    tokenizer = "tokenizer3"
    callback = push_to_hub_callback(output_dir, tokenizer)
    assert isinstance(callback, PushToHubCallback)

    # Test case 4
    output_dir = "model4"
    tokenizer = "tokenizer4"
    callback = push_to_hub_callback(output_dir, tokenizer)
    assert isinstance(callback, PushToHubCallback)

    # Test case 5
    output_dir = "model5"
    tokenizer = "tokenizer5"
    callback = push_to_hub_callback(output_dir, tokenizer)
    assert isinstance(callback, PushToHubCallback)
