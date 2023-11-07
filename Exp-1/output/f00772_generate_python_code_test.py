from f00772_generate_python_code import *
def test_generate_python_code():
    model_type = 'Bert'
    expected_code = 'processor = AutoProcessor.from_pretrained("Bert")\nmodel = AutoModelForQuestionAnswering.from_pretrained("Bert")'
    assert generate_python_code(model_type) == expected_code

    model_type = 'GPT2'
    expected_code = 'Invalid model type'
    assert generate_python_code(model_type) == expected_code

    model_type = 'Bloom'
    expected_code = 'processor = AutoProcessor.from_pretrained("Bloom")\nmodel = AutoModelForQuestionAnswering.from_pretrained("Bloom")'
    assert generate_python_code(model_type) == expected_code
