from typing import *
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

def generate_caption(image_url):
    tokenizer = AutoTokenizer.from_pretrained('model_name')
    model = AutoModelForCausalLM.from_pretrained('model_name')
    config = AutoConfig.from_pretrained('model_name')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = tokenizer(image, return_tensors='pt').to(device)
    generated_ids = model.generate(**inputs)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0]
