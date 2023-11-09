from transformers import AutoModel


def visual_question_answering(image_data, input_text):
    """
    This function uses a pre-trained model from Hugging Face to answer questions based on the provided image.

    Args:
        image_data (np.array): The image data to be processed.
        input_text (str): The question text to be answered.

    Returns:
        str: The answer to the question.
    """
    model = AutoModel.from_pretrained('sheldonxxxx/OFA_model_weights')
    vqa_result = model(image_data, input_text)
    return vqa_result