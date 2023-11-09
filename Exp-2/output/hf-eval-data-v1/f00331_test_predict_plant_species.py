from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
# Convert the dataset to a pandas DataFrame
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_df[iris['feature_names']], iris_df['target'], test_size=0.2, random_state=42)

# Test the predict_plant_species function
predictions = predict_plant_species(X_test)
# Check if the predictions are not None
assert predictions is not None
# Check if the length of the predictions is equal to the length of the test set
assert len(predictions) == len(X_test)