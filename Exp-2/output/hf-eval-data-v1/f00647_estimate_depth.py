from transformers import pipeline
import torch

# Function to estimate depth of objects in an image using a pre-trained model
# from Hugging Face Transformers.
#
# Parameters:
# image_path (str): Path to the image file.
#
# Returns:
# estimated_depth (torch.Tensor): Estimated depth map of the image.
def estimate_depth(image_path):
    # Create a depth estimation model using the pipeline function from the transformers library.
    # The model 'sayakpaul/glpn-nyu-finetuned-diode-221122-030603' is specified to be loaded.
    # This model is trained for depth estimation tasks.
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221122-030603')

    # Use the created depth estimation model to process the image and create a depth map.
    estimated_depth = depth_estimator(image_path)

    return estimated_depth