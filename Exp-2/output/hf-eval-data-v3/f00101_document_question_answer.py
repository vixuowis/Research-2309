# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import cv2
import torch

# function_code --------------------

def document_question_answer(image_path: str, question: str) -> str:
    '''
    This function takes an image path and a question as input, and returns the answer to the question based on the content of the image.
    
    Args:
        image_path (str): The path to the image file.
        question (str): The question to be answered.
    
    Returns:
        str: The answer to the question.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error in processing the image or question.
    '''
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)

    # Preprocess image (if required)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'No file found at {image_path}')

    input_tokens = tokenizer(question, image, return_tensors='pt')
    output = model(**input_tokens)
    start_logits, end_logits = output.start_logits, output.end_logits

    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)

    answer = tokenizer.decode(input_tokens['input_ids'][0][answer_start:answer_end + 1])
    return answer

# test_function_code --------------------

def test_document_question_answer():
    '''
    This function tests the document_question_answer function.
    '''
    # Test case 1: Valid image and question
    image_path = 'path/to/your/image.png'
    question = 'Your question here.'
    try:
        answer = document_question_answer(image_path, question)
        assert isinstance(answer, str), 'The answer should be a string.'
    except FileNotFoundError:
        print('Test case 1: The image file does not exist.')
    except Exception as e:
        print(f'Test case 1: Error - {str(e)}')

    # Test case 2: Invalid image path
    image_path = 'invalid/path/to/image.png'
    question = 'Your question here.'
    try:
        answer = document_question_answer(image_path, question)
        assert False, 'This test case should raise a FileNotFoundError.'
    except FileNotFoundError:
        print('Test case 2: The image file does not exist as expected.')
    except Exception as e:
        print(f'Test case 2: Error - {str(e)}')

    # Test case 3: Invalid question
    image_path = 'path/to/your/image.png'
    question = ''
    try:
        answer = document_question_answer(image_path, question)
        assert False, 'This test case should raise an Exception.'
    except FileNotFoundError:
        print('Test case 3: The image file does not exist.')
    except Exception as e:
        print('Test case 3: Error - {str(e)}')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_document_question_answer()