from f00795_load_model_multi_gpu import *
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_load_model_multi_gpu():
    model_name = "bigscience/bloom-2b5"
    model_8bit = load_model_multi_gpu(model_name)
    assert isinstance(model_8bit, AutoModelForCausalLM)
    assert model_8bit.device.type == 'cuda'
    assert model_8bit.device.index is not None
    assert model_8bit.load_in_8bit
    assert model_8bit.config.quantized

    # Test forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).cuda()
    outputs = model_8bit(input_ids=input_ids)
    assert 'logits' in outputs
    assert outputs.logits.shape == torch.Size([1, 5, model_8bit.config.vocab_size])
    assert torch.allclose(F.softmax(outputs.logits, dim=-1).sum(dim=-1), torch.ones(1, 5))


test_load_model_multi_gpu()
