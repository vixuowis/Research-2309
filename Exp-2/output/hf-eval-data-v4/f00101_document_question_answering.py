# requirements_file --------------------

!pip install -U transformers>=4.11.0 cv2 

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import cv2
import torch

# function_code --------------------

def document_question_answering(image_path, question):
    # Pre-trained model checkpoint
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)

    # Load and preprocess the image
    image = cv2.imread(image_path)

    # Tokenize the input
    input_tokens = tokenizer(question, image, return_tensors='pt')

    # Get the model output
    output = model(**input_tokens)
    start_logits, end_logits = output.start_logits, output.end_logits

    # Extract the predicted answer
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)
    answer = tokenizer.decode(input_tokens['input_ids'][0][answer_start:answer_end + 1])

    return answer

# test_function_code --------------------

# test_document_question_answering function assumes the existence of 'sample_image.png' as a sample document
# and 'what is the date today?' as a sample question
def test_document_question_answering():
    print('Testing started.')

    # Test case: Default
    print('Testing case [1/1] started.')
    predicted_answer = document_question_answering('sample_image.png', 'what is the date today?')
    assert predicted_answer is not None, f'Test case [1/1] failed: The function did not return an answer.'

    print('Testing finished.')

test_document_question_answering()