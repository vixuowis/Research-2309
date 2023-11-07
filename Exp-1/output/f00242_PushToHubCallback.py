from typing import *
from transformers.keras_callbacks import PushToHubCallback

def PushToHubCallback(output_dir: str, tokenizer: tokenizer) -> PushToHubCallback:
    push_to_hub_callback = PushToHubCallback(
        output_dir="my_awesome_wnut_model",
        tokenizer=tokenizer,
    )
