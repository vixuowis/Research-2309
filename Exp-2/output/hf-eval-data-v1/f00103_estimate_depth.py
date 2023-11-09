from transformers import pipeline
import torch

# Function to estimate depth from an image using a pre-trained model
# from Hugging Face Transformers.
# The model used is 'sayakpaul/glpn-nyu-finetuned-diode-221215-093747',
# which is a depth estimation model fine-tuned on the DIODE dataset.
# The function takes as input the path to an image and returns the depth map.
def estimate_depth(image_path):
    # Load the pre-trained model
    depth_estimator = pipeline('cv-depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221215-093747')
    # Estimate the depth map from the image
    depth_map = depth_estimator(image_path)
    return depth_map