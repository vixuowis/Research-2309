from f00504_get_predicted_label import *
def test_get_predicted_label():
    logits = torch.tensor([[0.1, 0.2, 0.7]])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label3'

    logits = torch.tensor([[0.9, 0.05, 0.05]])
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label1'

    logits = torch.tensor([[0.3, 0.4, 0.3]])
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label2'

test_get_predicted_label()
