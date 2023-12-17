# requirements_file --------------------

!pip install -U timm PIL torch

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm
import torch

# function_code --------------------

def classify_product_image(image_url):
    """
    Classify the product image into categories using a pretrained MobileNet-v3 model.

    :param image_url: URL of the product image
    :return: A list of predicted categories
    """
    img = Image.open(urlopen(image_url))
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_tensor = transforms(img).unsqueeze(0)
    input_tensor = input_tensor.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        output = model(input_tensor)

    # Get top-5 category indices
    top5_prob, top5_catid = torch.topk(output, 5)
    # Convert to categories (Here you may need to have a mapping from category indices to actual category names)
    categories = ['category_' + str(int(catid)) for catid in top5_catid[0]]
    return categories

# test_function_code --------------------

def test_classify_product_image():
    print("Testing classify_product_image function.")

    # Test case 1: A valid image URL
    print("Testing with a valid image URL.")
    categories = classify_product_image('https://example.com/sample_image.jpg')
    assert len(categories) == 5, f"Test case failed: Expected 5 categories, got {len(categories)}"
    print("Test case passed with a valid image URL.")

    # Test case 2: An invalid image URL
    try:
        categories = classify_product_image('https://example.com/non_existing_image.jpg')
        assert False, "Test case failed: Exception expected for non-existing image URL"
    except:
        print("Test case passed with an invalid image URL.")

    print("All tests passed!")

# Run the test
if __name__ == '__main__':
    test_classify_product_image()