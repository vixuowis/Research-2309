# requirements_file --------------------

!pip install -U keras-io/tab_transformer, sklearn

# function_import --------------------

from keras.tab_transformer.TabTransformer import TabTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# function_code --------------------

def customer_segmentation_with_tab_transformer(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the TabTransformer model from the provided configuration
    tab_transformer = TabTransformer.from_config()

    # Fit the model on the training data
    tab_transformer.fit(X_train, y_train)

    # Predict the segments for the test data
    predictions = tab_transformer.predict(X_test)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

# test_function_code --------------------

def test_customer_segmentation_with_tab_transformer():
    print("Testing started.")
    # Load a sample dataset
    # This should be replaced with an actual dataset loading function
    X, y = load_sample_data()

    print("Testing customer segmentation with TabTransformer.")
    accuracy, predictions = customer_segmentation_with_tab_transformer(X, y)

    # Define a threshold for acceptable accuracy
    acceptable_accuracy = 0.70

    assert accuracy > acceptable_accuracy, f"Test failed: Model accuracy {accuracy} is below {acceptable_accuracy}"
    assert len(predictions) > 0, "Test failed: No predictions made"
    print("Testing finished with accuracy:", accuracy)

# Helper function to generate sample data
# This should be replaced with an actual data loading function
def load_sample_data():
    import numpy as np
    X = np.random.rand(100, 10)  # Sample features
    y = np.random.randint(0, 3, 100)  # Sample labels for three segments
    return X, y

# Run the test
test_customer_segmentation_with_tab_transformer()