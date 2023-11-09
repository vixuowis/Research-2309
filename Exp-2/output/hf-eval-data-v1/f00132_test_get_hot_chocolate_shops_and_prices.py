def test_get_hot_chocolate_shops_and_prices():
    # Test data
    table = [["Shop", "Drink", "Price"], ["Cafe A", "Coffee", "3.00"], ["Cafe B", "Tea", "2.50"], ["Cafe C", "Hot Chocolate", "4.50"], ["Cafe D", "Hot Chocolate", "3.75"]]
    queries = ["Which shops sell hot chocolate and what are their prices?"]

    # Expected output
    expected_output = {"Cafe C": "4.50", "Cafe D": "3.75"}

    # Call the function with the test data
    output = get_hot_chocolate_shops_and_prices(table, queries)

    # Assert that the output is as expected
    assert set(output.keys()) == set(expected_output.keys())
    for shop in output:
        assert abs(float(output[shop]) - float(expected_output[shop])) < 0.01

test_get_hot_chocolate_shops_and_prices()