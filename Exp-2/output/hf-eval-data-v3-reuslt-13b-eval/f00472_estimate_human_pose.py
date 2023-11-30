# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
import os

# function_code --------------------

def estimate_human_pose(image_path: str, output_path: str) -> None:
    """
    Estimate the human pose from an image of a user performing an exercise.

    Args:
        image_path (str): The path to the image file.
        output_path (str): The path to save the output image with estimated pose.

    Returns:
        None
    """
    # load model and detector
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    diffuser_model_path = os.getenv("DIFFUSER_MODEL")
    controlnet_model_path = os.getenv("CONTROLNET_MODEL")
    model = ControlNetModel(diffuser_model_path, controlnet_model_path).to(device)
    detector = OpenposeDetector()
    
    # preprocess image
    input_image = Image.open(image_path)

    try:
        pose_data = detector.run_on_input(input_image, True)[-1]
        
        # draw pose on the image
        output_img = input_image.copy()
        for key in ["left_wrist", "right_wrist"]:
            x = int(pose_data[key][0])
            y = int(pose_data[key][1])
            output_img.putpixel((x,y), (256, 0, 0))

        # estimate the pose
        diffuser_pipeline = StableDiffusionControlNetPipeline()
        scheduler = UniPCMultistepScheduler(model)
        
        output_image = diffuser_pipeline.infer(scheduler, torch.from_numpy(pose_data["keypoints"][None].transpose((2,0,1))).to("cuda"), 150)[-1]
            
        # save result image to disk
        output_image = (output_image[-1].detach().cpu()*256).numpy()
        Image.fromarray(np.flipud(output_image)).save(output_path)
    except Exception as e:
        print("Could not estimate pose from image {}: {}".format(image_path, str(e)))

# test_function_code --------------------

def test_estimate_human_pose():
    """
    Test the function estimate_human_pose.
    """
    # Test case 1
    estimate_human_pose('test_images/exercise1.jpg', 'test_output/pose1_out.png')
    assert os.path.exists('test_output/pose1_out.png')

    # Test case 2
    estimate_human_pose('test_images/exercise2.jpg', 'test_output/pose2_out.png')
    assert os.path.exists('test_output/pose2_out.png')

    # Test case 3
    estimate_human_pose('test_images/exercise3.jpg', 'test_output/pose3_out.png')
    assert os.path.exists('test_output/pose3_out.png')

    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_human_pose()