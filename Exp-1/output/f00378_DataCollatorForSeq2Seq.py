from typing import *
from transformers import DataCollatorForSeq2Seq

def DataCollatorForSeq2Seq(tokenizer, model):
    """Data collator used for sequence-to-sequence tasks. This class dynamically pads the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

    Args:
    - tokenizer (`PreTrainedTokenizer`): The tokenizer used for encoding the inputs.
    - model (`PreTrainedModel`): The model used for computing the model inputs and outputs."""
    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors='pt')
        labels = batch.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch['labels'] = labels
        return batch
