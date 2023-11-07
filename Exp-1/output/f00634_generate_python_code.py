from typing import *
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

def generate_python_code(example):
    processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

    image = Image.open(example['image_id'])
    question = example['question']

    # prepare inputs
    inputs = processor(image, question, return_tensors="pt")

    model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]
