from f00425_get_predicted_class import *
def test_get_predicted_class():
    logits = torch.tensor([0.1, 0.5, 0.4])
    assert get_predicted_class(logits) == 1
    
    logits = torch.tensor([0.9, 0.2, 0.3])
    assert get_predicted_class(logits) == 0
    
    logits = torch.tensor([0.3, 0.4, 0.7])
    assert get_predicted_class(logits) == 2


if __name__ == '__main__':
    test_get_predicted_class()
