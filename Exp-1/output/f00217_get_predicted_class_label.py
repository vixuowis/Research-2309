from typing import *
from transformers import PreTrainedModel


def get_predicted_class_label(logits: torch.Tensor, model: PreTrainedModel) -> str:
    """
    Get the class with the highest probability, and use the model's `id2label` mapping to convert it to a text label:

    :param logits: The predicted logits from the model
    :type logits: torch.Tensor
    :param model: The pre-trained model
    :type model: PreTrainedModel
    :return: The predicted text label
    :rtype: str
    """
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]
