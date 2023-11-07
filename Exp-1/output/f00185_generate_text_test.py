from f00185_generate_text import *
model_inputs = {
    'input_ids': [[15496, 11, 616, 287, 13, 102]],
    'attention_mask': [[1, 1, 1, 1, 1, 1]]
}

def test_generate_text():
    generated_text = generate_text(model_inputs)
    assert isinstance(generated_text, str)
    print(generated_text)

test_generate_text()
