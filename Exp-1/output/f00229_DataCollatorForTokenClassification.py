from typing import *
from transformers import DataCollatorForTokenClassification

def DataCollatorForTokenClassification(tokenizer):
    """A data collator for token classification tasks."""
    """This data collator is specifically designed for token classification tasks. It takes a tokenizer and a list of token classification datasets (i.e., datasets containing input and label tokens) and dynamically pads the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

    Args:
    - tokenizer (`PreTrainedTokenizer`): The tokenizer used for encoding the input tokens.

    Returns:
    - `DataCollatorWithPadding`: The data collator with padding for token classification tasks."""
    return DataCollatorWithPadding(tokenizer)
