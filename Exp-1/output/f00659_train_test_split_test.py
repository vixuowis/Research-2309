from f00659_train_test_split import *
import numpy as np

# Test Case 1
dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
train_set, test_set = train_test_split(0.2)
assert len(train_set) == 3
assert len(test_set) == 1

# Test Case 2
dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
train_set, test_set = train_test_split(0.5)
assert len(train_set) == 2
assert len(test_set) == 2

# Test Case 3
dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
train_set, test_set = train_test_split(0.8)
assert len(train_set) == 1
assert len(test_set) == 3
