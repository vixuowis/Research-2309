from f00618_layoutlmv2_inference import *
def test_layoutlmv2_inference():
    question = "Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?"
    image = "path/to/image.jpg"

    expected_answer = "TRRF Vice President"

    assert layoutlmv2_inference(question, image) == expected_answer

    question = "What is the name of the person in the image?"

    expected_answer = "lee a. waller"

    assert layoutlmv2_inference(question, image) == expected_answer
