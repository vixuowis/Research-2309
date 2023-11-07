from f00450_get_predicted_label import *
def test_get_predicted_label():
    logits = torch.tensor([0.1, 0.3, 0.6])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label3'
    
    logits = torch.tensor([0.7, 0.2, 0.1])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label1'
    
    logits = torch.tensor([0.3, 0.5, 0.2])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label2'
    
    logits = torch.tensor([0.4, 0.1, 0.5])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label3'
    
    logits = torch.tensor([0.5, 0.1, 0.4])
    id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
    predicted_label = get_predicted_label(logits, id2label)
    assert predicted_label == 'label1'
    
    print('All test cases pass')

test_get_predicted_label()
