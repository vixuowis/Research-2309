from f00545_generate_python_code import *
def test_generate_python_code():
    code = generate_python_code("my_model", "path/to/model")
    expected_code = '>>> trainer = Trainer.from_pretrained("my_model", "path/to/model")\n>>> trainer.push_to_hub()'
    assert code == expected_code, 'Test failed'

test_generate_python_code()
