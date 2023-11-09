from transformers import pipeline


def get_image_caption(image_path: str, question: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to perform visual question answering tasks in the Polish language.
    It takes an image path and a question as input, processes the image using the model, and returns an answer to the question.

    Args:
        image_path (str): The path to the image that needs to be processed.
        question (str): The question that needs to be answered about the image.

    Returns:
        str: The answer to the question about the image.
    """
    vqa_pipeline = pipeline('visual-question-answering', model='azwierzc/vilt-b32-finetuned-vqa-pl')
    answer = vqa_pipeline(image_path, question)
    return answer