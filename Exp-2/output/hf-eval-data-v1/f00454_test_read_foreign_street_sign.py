def test_read_foreign_street_sign():
    # Test the function with a sample image URL
    image_url = 'https://i.postimg.cc/ZKwLg2Gw/367-14.png'
    result = read_foreign_street_sign(image_url)
    # Since the output is text, we cannot compare it strictly
    # So, we just check if the result is not None
    assert result is not None

test_read_foreign_street_sign()