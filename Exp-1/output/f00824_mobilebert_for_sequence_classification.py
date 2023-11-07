from typing import *
from transformers.models.mobilebert.modeling_mobilebert import MobileBertModel, MobileBertPreTrainedModel

class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    """
    MobileBert Model for sequence classification
    """

    def __init__(self, config):
