from f00453_train_test_split import *
import numpy as np

# Test case 1
train_set, test_set = train_test_split(0.2)
assert isinstance(train_set, np.ndarray)
assert isinstance(test_set, np.ndarray)

# Test case 2
train_set, test_set = train_test_split(0.3)
assert isinstance(train_set, np.ndarray)
assert isinstance(test_set, np.ndarray)

# Test case 3
train_set, test_set = train_test_split(0.1)
assert isinstance(train_set, np.ndarray)
assert isinstance(test_set, np.ndarray)
