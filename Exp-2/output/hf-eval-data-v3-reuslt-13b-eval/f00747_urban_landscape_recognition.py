# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def urban_landscape_recognition(image_url):
    """
    Recognize urban landscapes and identify different objects in the image using SegformerForSemanticSegmentation model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        logits (torch.Tensor): The output logits from the model which can be used to identify different objects in the image.

    Raises:
        OSError: If there is a problem with the disk quota or the file handling.
    """  # noqa

    try:
        # load segformer_humanseg_mobile from huggingface hub
        extractor = SegformerFeatureExtractor(size=512, do_padding=False)
        model = SegformerForSemanticSegmentation.from_pretrained('nicolalandro/segformer_humanseg_mobile').to("cuda")
        
        # download the image and prepare it for inference
        response = requests.get(image_url, timeout=4)
        if response.status_code != 200:
            raise Exception('Image cannot be accessed')
        else:
            input_image = Image.open(BytesIO(response.content))
            inputs = extractor(images=input_image, return_tensors="pt")
        
        # run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # get the predicted pixel labels 
            
    except OSError as e:
        print("OSError while processing image: ", repr(e))
    
    return logits

# test_function_code --------------------

def test_urban_landscape_recognition():
    """
    Test the function urban_landscape_recognition.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    logits = urban_landscape_recognition(image_url)
    assert logits is not None, 'The output logits should not be None.'
    assert logits.shape[0] == 1, 'The first dimension of the output logits should be 1.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_urban_landscape_recognition()