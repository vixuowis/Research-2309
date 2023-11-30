# function_import --------------------

from transformers import DPTForDepthEstimation
import numpy as np

# function_code --------------------

def estimate_depth(model_name: str, drone_footage: np.ndarray) -> np.ndarray:
    """
    Estimate the depth in drone footage using a pre-trained DPTForDepthEstimation model.

    Args:
        model_name (str): The name of the pre-trained DPTForDepthEstimation model.
        drone_footage (np.ndarray): The drone footage to estimate depth from.

    Returns:
        np.ndarray: The estimated depth map.
    """    
    # Initialize model and set on cuda if available
    model = DPTForDepthEstimation.from_pretrained(model_name).cuda()
    model.eval()
    # Convert to tensor and normalize
    x = (255*drone_footage).astype(np.float32)
    for i in range(x.shape[0]): 
        x[i,:] = (x[i,:]-x[i,:].mean())/x[i,:].std()
    x = np.transpose(x,(0,3,1,2))
    x_tensor = torch.as_tensor(x).type(torch.FloatTensor)
    # Estimate depth
    with torch.no_grad():
        output = model(x_tensor.cuda()).squeeze()
    output = torch.nn.functional.interpolate(output, (360, 640), mode='bilinear', align_corners=True)
    
    # Convert back to numpy and post-processes
    depth = output.cpu().numpy()*175
    depth = depth[:,:,68:292]
    depth = np.transpose(depth,(0,2,3,1)).astype(np.float32)
    
    # Remove values outside of range and set nan's to 0.0
    depth[depth > 75.0]=75.0
    depth[depth < 1.0] = 1.0
    depth[np.isnan(depth)] = 0.0
    
    return depth

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    model_name = 'hf-tiny-model-private/tiny-random-DPTForDepthEstimation'
    drone_footage = np.random.rand(100, 100, 3)
    depth_map = estimate_depth(model_name, drone_footage)
    assert depth_map.shape == drone_footage.shape, 'The shape of the depth map should be the same as the drone footage.'
    assert np.all(depth_map == 0), 'The depth map should be all zeros for this placeholder implementation.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_depth()