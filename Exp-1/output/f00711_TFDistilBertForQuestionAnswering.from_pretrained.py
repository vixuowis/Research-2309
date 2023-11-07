from typing import *
from transformers import TFDistilBertForQuestionAnswering

tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
