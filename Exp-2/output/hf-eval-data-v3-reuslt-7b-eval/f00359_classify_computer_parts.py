# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_computer_parts(user_uploaded_image_file_path):
    '''
    Classify the computer parts in the image uploaded by the user.

    Args:
        user_uploaded_image_file_path (str): The file path of the image uploaded by the user.

    Returns:
        str: The predicted label of the computer part in the image.
    '''

    # Load pre-trained model and tokenizer for ViT for image classification. 
    # For more details, please refer to https://huggingface.co/google/vit-base-patch16-224-in21k .
    processor = ViTImageProcessor(feature_extractor="google/vit-base-patch16-224-in21k",
                                  return_tensors='pt')
    
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

    # Load image and convert into the feature tensors using the pre-trained processor. 
    image = Image.open(user_uploaded_image_file_path)
    
    encoding = processor(images=image, return_tensors="pt", padding=True)
    
    # Predict labels for the image uploaded by user.
    outputs = model(**encoding['pixel_values']) 

    # Print out the predicted label and confidence score in percentage.
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_probability = logits.softmax(-1)[0, predicted_class_idx] * 100
    print('Predicted class:', model.config.id2label[predicted_class_idx])
    print(f'Probability: {predicted_class_probability:.2f}%')
    
# main --------------------

if __name__ == '__main__':
    # Download sample image from the internet for testing purpose.
    response = requests.get('https://raw.githubusercontent.com/prasadp51/vit-pytorch/master/images/sample_image.png') 
    file_path = 'tempfile'
    open(f'{file_path}.jpg', 'wb').write(response.content)
    
    # Call function for predicting the computer part in sample image.  
    classify_computer_parts(user_upload

# test_function_code --------------------

def test_classify_computer_parts():
    '''
    Test the function classify_computer_parts.
    '''
    url = 'https://placekitten.com/200/300'
    response = requests.get(url, stream=True)
    with open('test_image.jpg', 'wb') as f:
        f.write(response.content)

    predicted_label = classify_computer_parts('test_image.jpg')
    assert isinstance(predicted_label, str), 'The predicted label should be a string.'

    print('All Tests Passed')


# call_test_function_code --------------------

test_classify_computer_parts()