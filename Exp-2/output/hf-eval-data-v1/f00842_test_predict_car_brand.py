def test_predict_car_brand():
    """
    This function tests the predict_car_brand function.
    It uses a sample image of a car and checks if the predicted brand is in the list of known car brands.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    predicted_brand = predict_car_brand(url)
    known_brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Volkswagen', 'Subaru']
    assert predicted_brand in known_brands, f'Unexpected brand: {predicted_brand}'