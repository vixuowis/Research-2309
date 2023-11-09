def test_generate_product_description():
    """
    This function tests the generate_product_description function.
    It uses a sample image from the COCO dataset, as the model was fine-tuned on this dataset.
    The generated description is not compared strictly, as the model can generate different descriptions for the same image.
    """
    # Load the sample image
    sample_image = 'path_to_sample_image'
    
    # Generate the product description
    product_description = generate_product_description(sample_image)
    
    # Check if the product description is a string
    assert isinstance(product_description, str), 'The product description should be a string.'
    
    # Check if the product description is not empty
    assert product_description, 'The product description should not be empty.'

test_generate_product_description()