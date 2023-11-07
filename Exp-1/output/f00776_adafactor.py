from typing import *
from transformers import Adafactor

def adafactor(training_args, per_device_train_batch_size, **default_args):
    """
    Adafactor doesn't store rolling averages for each element in weight matrices. Instead, it keeps aggregated information
    (sums of rolling averages row- and column-wise), significantly reducing its footprint. However, compared to Adam,
    Adafactor may have slower convergence in certain cases.

    You can switch to Adafactor by setting `optim="adafactor"` in [`TrainingArguments`]:
    """
    training_args = TrainingArguments(per_device_train_batch_size=per_device_train_batch_size, optim="adafactor", **default_args)
