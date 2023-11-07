from f00424_generate_logits import *
def test_generate_logits():
    model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
    inputs = {}
    labels = torch.tensor([0, 1, 2, 3, 4])
    logits = generate_logits(model, inputs, labels)
    assert logits.shape == (5, num_choices)
    assert torch.all(logits >= 0) and torch.all(logits <= 1)
    print("All tests pass.")

test_generate_logits()
