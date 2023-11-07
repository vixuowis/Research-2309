from typing import *
from transformers import TFAutoModelForSemanticSegmentation

def load_segformer(checkpoint, id2label, label2id, optimizer):
    '''
    Load SegFormer with TFAutoModelForSemanticSegmentation along with the label mappings, and compile it with the optimizer.
    
    Args:
    - checkpoint (str): The path or name of the checkpoint to load the model from.
    - id2label (dict): A dictionary mapping class IDs to class labels.
    - label2id (dict): A dictionary mapping class labels to class IDs.
    - optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training.
    
    Returns:
    - model (TFAutoModelForSemanticSegmentation): The loaded and compiled SegFormer model.
    '''
    model = TFAutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
    model.compile(optimizer=optimizer)
