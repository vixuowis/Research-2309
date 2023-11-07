from typing import *
from transformers.keras_callbacks import PushToHubCallback

push_to_hub_callback = PushToHubCallback(
    output_dir="my_awesome_model",
    tokenizer=tokenizer,
)
