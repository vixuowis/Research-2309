from f00018_apply_softmax import *
def test_apply_softmax():
	# Test case 1
	logits1 = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
	expected1 = torch.tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364], [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]])
	assert torch.allclose(apply_softmax(logits1), expected1)

	# Test case 2
	logits2 = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
	expected2 = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]])
	assert torch.allclose(apply_softmax(logits2), expected2)

	# Test case 3
	logits3 = torch.tensor([[1, 2, 3, 4, 5]])
	expected3 = torch.tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364]])
	assert torch.allclose(apply_softmax(logits3), expected3)

	print('All test cases pass')

test_apply_softmax()
