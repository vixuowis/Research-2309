def test_estimate_pool_depth():
    """
    This function tests the 'estimate_pool_depth' function.
    It uses a sample underwater photo and checks if the returned depth is within a reasonable range.
    """
    # Sample underwater photo
    sample_photo = 'sample_underwater_photo.jpg'
    
    # Get depth estimation
    estimated_depth = estimate_pool_depth(sample_photo)
    
    # Check if the estimated depth is within a reasonable range
    assert 0 <= estimated_depth <= 50, 'The estimated depth should be between 0 and 50 meters.'