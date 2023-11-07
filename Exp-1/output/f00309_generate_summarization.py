from typing import *
from transformers import TFAutoModelForCausalLM

def generate_summarization(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
    """Generate a summarization using the TFAutoModelForCausalLM model.

    Args:
        input_ids (tf.Tensor): The input token IDs.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
        do_sample (bool, optional): Whether to use sampling during generation. Defaults to True.
        top_k (int, optional): The number of highest probability tokens to keep during sampling. Defaults to 50.
        top_p (float, optional): The cumulative probability threshold for top-k and top-p sampling. Defaults to 0.95.

    Returns:
        tf.Tensor: The generated summarization.
    """
    model = TFAutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
    return outputs

