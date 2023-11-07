from typing import *
from transformers import TrainingArguments

def enable_gradient_accumulation(training_args: TrainingArguments, accumulation_steps: int) -> TrainingArguments:
    
    training_args.gradient_accumulation_steps = accumulation_steps
    return training_args
