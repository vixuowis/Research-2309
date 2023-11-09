from transformers import AutoModelForQuestionAnswering


def load_question_answering_model(model_name):
    """
    This function loads a question answering model from the transformers library.
    
    Parameters:
    model_name (str): The name of the model to load.
    
    Returns:
    model: The loaded model.
    """
    # Load the model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Return the model
    return model