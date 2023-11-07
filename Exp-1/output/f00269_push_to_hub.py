from typing import *
from transformers.keras_callbacks import PushToHubCallback

def push_to_hub(output_dir: str, tokenizer: PreTrainedTokenizer) -> PushToHubCallback:
    callback = PushToHubCallback(
        output_dir=output_dir,
        tokenizer=tokenizer,
    )
