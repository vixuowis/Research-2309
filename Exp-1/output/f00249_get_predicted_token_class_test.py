from f00249_get_predicted_token_class import *
def test_get_predicted_token_class():
    logits = torch.tensor([[[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9]]])
    id2label = {0: 'O', 1: 'B-location', 2: 'I-location'}
    expected_output = ['O', 'O', 'B-location']
    output = get_predicted_token_class(logits, id2label)
    assert output == expected_output, f'Expected: {expected_output}, Got: {output}'

test_get_predicted_token_class()
