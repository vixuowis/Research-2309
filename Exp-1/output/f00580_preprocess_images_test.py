from f00580_preprocess_images import *
def test_preprocess_images():
    image_target = ["image1.jpg", "image2.jpg", "image3.jpg"]
    query_images = ["query1.jpg", "query2.jpg", "query3.jpg"]
    
    result = preprocess_images(image_target, query_images)
    
    assert len(result) == 2
    assert "pixel_values" in result
    assert "query_pixel_values" in result
    assert result["pixel_values"].shape == (3, 3, 224, 224, 3)
    assert result["query_pixel_values"].shape == (3, 224, 224, 3)
    
    print("All tests pass.")
    
test_preprocess_images()
