from f00061_iterate_dataset import *
def test_iterate_dataset():
	# Create a dummy dataset
	dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 2, (10,)))
	# Iterate over the dataset
	for batch in iterate_dataset(dataset, batch_size=2):
		print(batch)

test_iterate_dataset()
