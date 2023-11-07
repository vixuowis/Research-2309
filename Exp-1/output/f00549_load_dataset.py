from typing import *
from datasets import load_dataset

def load_cppe5_dataset():
    '''
    Load the CPPE-5 dataset

    Returns:
        cppe5 (DatasetDict): The CPPE-5 dataset
    '''
    cppe5 = load_dataset('cppe-5')
    return cppe5
