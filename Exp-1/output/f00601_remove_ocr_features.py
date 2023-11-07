from typing import *
from datasets import Dataset

def remove_ocr_features(dataset: Dataset) -> Dataset:
    '''Removes OCR features from the dataset.'''
    updated_dataset = dataset.remove_columns('words')
    updated_dataset = updated_dataset.remove_columns('bounding_boxes')
    return updated_dataset
