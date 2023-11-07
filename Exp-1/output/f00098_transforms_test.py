from f00098_transforms import *
def test_transforms():
    examples = {
        "image": [Image.open("image1.jpg"), Image.open("image2.jpg"), Image.open("image3.jpg")]
    }
    transformed_examples = transforms(examples)
    assert "pixel_values" in transformed_examples
    assert transformed_examples["pixel_values"].shape[0] == len(examples["image"])
    print("Transforms function passed all test cases.")

test_transforms()
