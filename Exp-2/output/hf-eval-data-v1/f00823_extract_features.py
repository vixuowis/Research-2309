from transformers import BertTokenizer, AutoModel
import torch

def extract_features(input_text):
    """
    This function is used to extract features from Indonesian text using the IndoBERT model.

    Args:
        input_text (str): The Indonesian text from which to extract features.

    Returns:
        torch.Tensor: The contextual representation of the input text.
    """
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    contextual_representation = model(encoded_input)[0]
    return contextual_representation