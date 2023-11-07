from f00251_get_token_classification_logits import *
from transformers import TFAutoModelForTokenClassification
import numpy as np

model = TFAutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")

inputs = {
	"input_ids": np.array([[101, 2054, 2003, 1037, 2518, 2000, 2022, 1996, 2307, 1012, 102], [101, 2054, 2003, 1037, 2518, 2000, 2022, 1996, 2307, 1012, 102]]),
	"attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}

logits = get_token_classification_logits(model, inputs)
print(logits)
