from f00501_run_image_classification import *
def test_run_image_classification():
    model_name = "my_awesome_food_model"
    image = "path/to/image.jpg"
    result = run_image_classification(model_name, image)
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "score" in result[0]
    assert "label" in result[0]
