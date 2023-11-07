from f00637_generate_text import *
def test_generate_text():
    image = PIL.Image.open("image.jpg")
    prompt = "Please describe the image."
    processor = SomeProcessor()
    model = SomeModel()
    device = torch.device("cuda")

    generated_text = generate_text(image, prompt, processor, model, device)

    assert generated_text == "He is looking at the crowd"

test_generate_text()
