from f00824_mobilebert_for_sequence_classification import *
def test_mobilebert_for_sequence_classification():
    model = MobileBertForSequenceClassification(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs[0].size() == (1, config.hidden_size)
    assert outputs[1].size() == (1, config.hidden_size)
    assert outputs[2].size() == (1, 2)
    assert torch.allclose(outputs[2].sum(-1), torch.ones(1))
    print("Test passed.")
