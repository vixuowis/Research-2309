from typing import *
from datasets import load_dataset

def load_billsum_dataset() -> dict:
    """Load BillSum dataset

    Start by loading the smaller California state bill subset of the BillSum dataset from the 🤗 Datasets library:

    :return: The loaded BillSum dataset
    """
    billsum = load_dataset("billsum", split="ca_test")
    return billsum
