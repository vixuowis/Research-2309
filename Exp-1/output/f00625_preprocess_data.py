from typing import *
from datasets import Dataset

def preprocess_data(example):
    # Preprocess the data
    # Remove unnecessary columns
    # Apply the preprocessing function to each element of the dataset
    # Set batched=True to process multiple elements of the dataset at once
    # Return the processed dataset
    example.pop('question')
    example.pop('question_type')
    example.pop('question_id')
    example.pop('image_id')
    example.pop('answer_type')
    example.pop('label.ids')
    example.pop('label.weights')

    return example
