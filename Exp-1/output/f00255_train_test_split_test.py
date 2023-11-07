from f00255_train_test_split import *
import assert

# Test case 1
dataset = Dataset()
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)
assert len(train_dataset) == 0.8 * len(dataset)
assert len(test_dataset) == 0.2 * len(dataset)

# Test case 2
dataset = Dataset()
train_dataset, test_dataset = dataset.train_test_split(test_size=0.5)
assert len(train_dataset) == 0.5 * len(dataset)
assert len(test_dataset) == 0.5 * len(dataset)

# Test case 3
dataset = Dataset()
train_dataset, test_dataset = dataset.train_test_split(test_size=0.1)
assert len(train_dataset) == 0.9 * len(dataset)
assert len(test_dataset) == 0.1 * len(dataset)
