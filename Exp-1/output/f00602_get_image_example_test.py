from f00602_get_image_example import *
def test_get_image_example():
    dataset = {"train": [{"image": "image1"}, {"image": "image2"}, {"image": "image3"}, {"image": "image4"}, {"image": "image5"}]}
    assert get_image_example(dataset, 0) == "image1"
    assert get_image_example(dataset, 2) == "image3"
    assert get_image_example(dataset, 4) == "image5"

    print("All test cases pass")


test_get_image_example()
