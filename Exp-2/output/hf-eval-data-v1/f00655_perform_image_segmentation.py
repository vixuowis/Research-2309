from some_module import UperNetModel


def perform_image_segmentation(image):
    """
    This function performs semantic segmentation on an input image using a pre-trained UperNet model.
    
    Parameters:
    image (Image): The input image to be segmented.
    
    Returns:
    Image: The segmented image.
    """
    # Load the pre-trained UperNet model
    model = UperNetModel.from_pretrained('openmmlab/upernet-convnext-small')
    
    # Perform semantic segmentation on the input image
    segmented_image = model(image)
    
    return segmented_image