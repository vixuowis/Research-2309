# requirements_file --------------------

import subprocess

requirements = ["keras", "sklearn"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from keras.tab_transformer import TabTransformer
from sklearn.model_selection import train_test_split

# function_code --------------------

def segment_customers_using_tabtransformer(X, y):
    """
    Segments customers into different categories using the TabTransformer model.

    Args:
        X (DataFrame): The input features in a pandas DataFrame including both numerical
                       and categorical data.
        y (Series): The target variable as a pandas Series.

    Returns:
        tuple: A tuple containing the trained TabTransformer model and predictions for the test set.

    Raises:
        ValueError: If the input data X is not a pandas DataFrame or y is not a pandas Series.
    """
    # Verify that input is pandas DataFrame and Series
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError('Input data X must be a pandas DataFrame and y must be a pandas Series.')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and configure TabTransformer model
    tab_transformer = TabTransformer.from_config()

    # Fit the model on training data
    tab_transformer.fit(X_train, y_train)

    # Generate predictions on the test set
    predictions = tab_transformer.predict(X_test)

    return tab_transformer, predictions

# test_function_code --------------------

def test_segment_customers_using_tabtransformer():
    print("Testing started.")
    # Load sample dataset
    X, y = load_sample_data()

    # Testing case 1: Proper DataFrame and Series inputs
    print("Testing case [1/2] started.")
    try:
        model, predictions = segment_customers_using_tabtransformer(X, y)
        assert len(predictions) > 0, "Test case [1/2] failed: no predictions made"
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Testing case 2: Improper input types
    print("Testing case [2/2] started.")
    try:
        model, predictions = segment_customers_using_tabtransformer(X.values, y.values)
        assert False, "Test case [2/2] failed: ValueError not raised for improper input types"
    except ValueError:
        pass  # Expecting a ValueError
    except Exception as e:
        assert False, f"Test case [2/2] failed with an unexpected exception: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_customers_using_tabtransformer()