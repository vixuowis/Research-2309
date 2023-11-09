from transformers import Pix2StructForConditionalGeneration, T5Tokenizer, T5Config


def load_pix2struct_model():
    """
    This function loads the Pix2Struct model for visual question answering from Hugging Face Transformers.
    The model is pretrained on the 'google/pix2struct-chartqa-base'.

    Returns:
        model: A Pix2StructForConditionalGeneration model.
    """
    config = T5Config.from_pretrained('google/pix2struct-chartqa-base')
    tokenizer = T5Tokenizer.from_pretrained('google/pix2struct-chartqa-base')
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-chartqa-base', config=config)
    return model