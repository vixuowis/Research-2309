from typing import *
from transformers import PushToHubCallback

push_to_hub_callback = PushToHubCallback(output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model")
