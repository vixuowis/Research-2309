from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def answer_recipe_question(question_text, recipe_image):
    """
    This function takes a question about a recipe and an image of the recipe as input, and returns an answer to the question.
    The function uses a pretrained model for visual question answering tasks.

    Args:
        question_text (str): The question about the recipe.
        recipe_image (str): The image of the recipe.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa')
    tokenizer = AutoTokenizer.from_pretrained('uclanlp/visualbert-vqa')
    inputs = tokenizer(question_text, recipe_image, return_tensors='pt')
    outputs = model(**inputs)
    answer = tokenizer.decode(outputs['start_logits'], outputs['end_logits'])
    return answer