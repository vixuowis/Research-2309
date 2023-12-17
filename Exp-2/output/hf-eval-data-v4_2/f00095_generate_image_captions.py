# requirements_file --------------------

!pip install -U transformers torch PIL

# function_import --------------------



# function_code --------------------

def generate_image_captions(image_paths):
    """
    Generates captions for a list of image paths using pre-trained VisionEncoderDecoder model.

    Args:
        image_paths: A list of paths to the image files.

    Returns:
        A list of captions for the images.

    Raises:
        FileNotFoundError: If any image path is not found.
    """

    import torch
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    from PIL import Image

    # Load the pre-trained model, feature extractor, and tokenizer
    model_name = 'nlpconnect/vit-gpt2-image-captioning'
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Generate captions
    captions = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            pixel_values = feature_extractor(images=[image], return_tensors='pt').pixel_values.to(device)
            output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            captions.append(caption)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Image file not found: {image_path}') from e
    return captions

# test_function_code --------------------

def test_generate_image_captions():
    from pathlib import Path
    print("Testing started.")

    # Prepare a list of image paths for testing
    test_image_paths = list(Path('test_images').glob('*.jpg'))[:3]
    expected_number_of_captions = len(test_image_paths)

    # Test case 1: Check if the correct number of captions is generated
    print("Testing case [1/3] started.")
    captions = generate_image_captions(test_image_paths)
    assert len(captions) == expected_number_of_captions, f"Test case [1/3] failed: Expected {expected_number_of_captions} captions, got {len(captions)}"

    # Further test cases, specific to the domain of image captioning, would ideally require human judgment, hence omitted.
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_captions()