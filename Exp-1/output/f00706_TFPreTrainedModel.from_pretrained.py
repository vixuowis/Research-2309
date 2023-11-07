from typing import *
from transformers import TFDistilBertModel

tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
