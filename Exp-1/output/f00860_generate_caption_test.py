from f00860_generate_caption import *
def test_generate_caption():
    text_prompt = "Generate a coco-style caption.\n"
    image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
    expected_caption = "A red bus driving down the street."

    generated_caption = generate_caption(text_prompt, image_url)

    assert generated_caption == expected_caption, f"Expected: {expected_caption}, but got: {generated_caption}"
