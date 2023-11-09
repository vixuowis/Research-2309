def test_estimate_human_pose():
    """
    Test the function estimate_human_pose.
    """
    import os
    image_path = 'test_image_path.png'
    text_prompt = 'actor performing a scene'
    num_inference_steps = 20
    estimate_human_pose(image_path, text_prompt, num_inference_steps)
    assert os.path.exists('images/actor_pose_out.png'), 'The output image does not exist.'