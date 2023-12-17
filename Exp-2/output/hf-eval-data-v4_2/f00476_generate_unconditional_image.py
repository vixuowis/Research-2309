# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------


    def generate_unconditional_image(model_id: str) -> Image:
        """
        Generate an unconditional image using a given DDPM model.

        Args:
            model_id (str): The model ID of the pretrained DDPM model.

        Returns:
            Image: The generated image as a PIL image object.

        Raises:
            ValueError: If the model_id is None or empty.

        """
        if not model_id:
            raise ValueError('model_id must not be None or empty.')
        ddpm = DDPMPipeline.from_pretrained(model_id)
        image = ddpm().images[0]
        return image


# test_function_code --------------------


    def test_generate_unconditional_image():
        print("Testing started.")
        model_id = 'google/ddpm-church-256'

        # 测试用例 1: 正确的 model_id
        print("Testing case [1/1] started.")
        image = generate_unconditional_image(model_id)
        assert image.size == (256, 256), f"Test case [1/1] failed: Expected image of size (256, 256), got {image.size}"
        print("Testing finished.")

# call_test_function_line --------------------

test_generate_unconditional_image()