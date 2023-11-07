from typing import *
from transformers import DataCollatorForSeq2Seq

def DataCollatorForSeq2Seq(tokenizer, model):
    """This class dynamically pads the sentences to the longest length in a batch during collation.

    Args:
        tokenizer (Tokenizer): The tokenizer used for tokenization.
        model (PreTrainedModel): The model used for training or inference.

    Returns:
        DataCollatorForSeq2Seq: The data collator object.
    """
    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors='pt')
        batch['labels'] = batch['input_ids']
        batch['decoder_input_ids'] = batch['input_ids']
        return batch
