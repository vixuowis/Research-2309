def test_predict_income_category():
    '''
    Test the function predict_income_category.
    '''
    # Load the dataset
    dataset = tf.data.experimental.CsvDataset('census_income_dataset.csv', header=True)

    # Select a sample from the dataset
    sample = next(iter(dataset.take(1)))

    # Convert the sample to a dictionary
    input_features = {key: value for key, value in zip(dataset.keys(), sample)}

    # Call the function with the sample
    result = predict_income_category(input_features)

    # Check the result
    assert isinstance(result, str), 'The result should be a string.'
    assert result in ['Over 50K per year.', '50K or less per year.'], 'The result should be one of the two income categories.'

test_predict_income_category()