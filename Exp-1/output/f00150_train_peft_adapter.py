from typing import *
from peft import LoraConfig


def train_peft_adapter():
    """Train a PEFT adapter

    PEFT adapters are supported by the [`Trainer`] class so that you can train an adapter for your specific use case. It only requires adding a few more lines of code. For example, to train a LoRA adapter:

    <Tip>

    If you aren't familiar with fine-tuning a model with [`Trainer`], take a look at the [Fine-tune a pretrained model](training) tutorial.

    </Tip>

    1. Define your adapter configuration with the task type and hyperparameters (see [`~peft.LoraConfig`] for more details about what the hyperparameters do).
    """
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
