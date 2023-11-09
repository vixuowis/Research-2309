from transformers import pipeline


def image_question_answering(image_path: str, question: str) -> str:
    """
    This function uses the 'uclanlp/visualbert-vqa' model from Hugging Face Transformers to answer questions related to the contents of images.
    
    Parameters:
    image_path (str): The path to the image file.
    question (str): The question related to the image.
    
    Returns:
    str: The answer to the question.
    """
    # Load the 'uclanlp/visualbert-vqa' model
    model = pipeline('question-answering', model='uclanlp/visualbert-vqa')
    
    # Provide the image and the question to the model
    result = model(image_path, question)
    
    # Return the answer
    return result