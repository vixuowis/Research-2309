from typing import *
from transformers import AutoModelForSequenceClassification

def train_with_pytorch_trainer():
    """
    Train with PyTorch Trainer

    🤗 Transformers provides a [`Trainer`] class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [`Trainer`] API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

    Start by loading your model and specify the number of expected labels. From the Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), you know there are five labels:
    """
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
