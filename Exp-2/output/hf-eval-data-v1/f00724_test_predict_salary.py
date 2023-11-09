def test_predict_salary():
    # Load the dataset
    dataset = pd.read_csv('Census-Income Data Set.csv')
    
    # Select a sample from the dataset
    sample = dataset.sample()
    
    # Predict the salary class of the sample
    prediction = predict_salary(sample)
    
    # Check if the prediction is close to the actual value
    assert abs(prediction - sample['salary'].values[0]) < 0.1