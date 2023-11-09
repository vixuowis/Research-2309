from keras.tab_transformer.TabTransformer import TabTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Function to classify customer segments using TabTransformer
# @param df: Input dataframe
# @param target: Target column name
# @return: Trained model and predictions

def classify_customer_segments(df, target):
    # Preprocessing
    # Encoding categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Splitting the dataset
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Loading the model
    tab_transformer = TabTransformer.from_config()
    
    # Training the model
    tab_transformer.fit(X_train, y_train)
    
    # Making predictions
    predictions = tab_transformer.predict(X_test)
    
    return tab_transformer, predictions