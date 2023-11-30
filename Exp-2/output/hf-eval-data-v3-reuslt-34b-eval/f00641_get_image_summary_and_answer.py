# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_image_summary_and_answer(img_url: str, question: str) -> str:
    """
    Get a text summary and answer a question from an image using the 'Blip2ForConditionalGeneration' model.

    Args:
        img_url (str): The URL of the image.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        Exception: If there is an error in processing the image or generating the answer.
    """
    
    # Image to text summary --------------------
    
    res = requests.get(img_url)
    
    try:
      img = Image.open(res.content).convert("RGB")
      
    except:
        raise Exception('There was a problem processing the image.')
      
    # Image captioning --------------------
    
    model = Blip2ForConditionalGeneration.from_pretrained('BruceWen/blip2-causal-image-prompt')
    processor = BlipProcessor.from_pretrained('BruceWen/blip2-causal-image-prompt')
    
    input_ids = processor(img, return_tensors='pt').pixel_values
    input_ids = input_ids.to('cuda')
    output_ids = model.generate(input_ids)
    gen = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Answer a question --------------------
    
    question = f"{gen} {question}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Blip2ForConditionalGeneration.from_pretrained("BruceWen/blip2-base-finetuned-q+a")
    processor = BlipProcessor.from_pretrained('BruceWen/blip2-base-finetuned-q+a')

    inputs = processor(question, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(**inputs)
    
    answer = processor.batch_decode(output_ids[0], skip_special_tokens=True)[0]
    
    # Remove the question from the answer --------------------
    
    answer_cleaned = answer.replace(gen,"")
    return answer_cleaned


# test_function_code --------------------

def test_get_image_summary_and_answer():
    """
    Test the function 'get_image_summary_and_answer'.
    """
    try:
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'What is the main color of the object?') is not None
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'Is there a cat in the image?') is not None
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'What is the size of the object?') is not None
        print('All Tests Passed')
    except Exception as e:
        print('Test Failed: ' + str(e))


# call_test_function_code --------------------

test_get_image_summary_and_answer()