from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test the predict_recidivism function
predictions = predict_recidivism(X_test)

# Check that the function returns the correct output type
assert isinstance(predictions, np.ndarray), 'Expected output type: numpy.ndarray'

# Check that the function returns the correct output shape
assert predictions.shape == y_test.shape, 'Expected output shape: {}'.format(y_test.shape)