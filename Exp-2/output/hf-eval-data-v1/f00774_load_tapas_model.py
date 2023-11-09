from transformers import TapasForQuestionAnswering


def load_tapas_model():
    """
    Load the pre-trained TAPAS model for table question answering.

    This function loads the TAPAS model that has been fine-tuned on the WikiSQL dataset.
    The model is provided by the Transformers library from Hugging Face.

    Returns:
        model: The loaded TAPAS model.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    return model