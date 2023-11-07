from f00242_PushToHubCallback import *
import assert

def test_PushToHubCallback():
    output_dir = "my_awesome_wnut_model"
    tokenizer = tokenizer
    push_to_hub_callback = PushToHubCallback(output_dir, tokenizer)
    assert push_to_hub_callback.output_dir == output_dir
    assert push_to_hub_callback.tokenizer == tokenizer

test_PushToHubCallback()
