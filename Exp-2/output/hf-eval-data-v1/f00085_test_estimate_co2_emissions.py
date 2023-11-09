def test_estimate_co2_emissions():
    # Load test data
    test_data = pd.read_csv('datadmg/autotrain-data-test-news.csv')
    # Select a sample from the test data
    sample = test_data.sample()
    # Get the sample's features
    engine_size = sample['engine_size'].values[0]
    transmission_type = sample['transmission_type'].values[0]
    miles_traveled = sample['miles_traveled'].values[0]
    # Estimate CO2 emissions for the sample
    estimated_emissions = estimate_co2_emissions(engine_size, transmission_type, miles_traveled)
    # Assert that the estimated emissions are within a reasonable range
    assert 0 <= estimated_emissions <= 1000, 'Estimated emissions are out of range'