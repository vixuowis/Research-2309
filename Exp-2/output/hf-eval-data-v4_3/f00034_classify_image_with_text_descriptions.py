# requirements_file --------------------

import subprocess

requirements = ["PIL", "requests", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_image_with_text_descriptions(image_url, texts):
    """
    Classifies an image based on provided text descriptions using a pre-trained Chinese CLIP model.

    Args:
        image_url (str): The URL of the image to be classified.
        texts (List[str]): A list of text descriptions for zero-shot classification.

    Returns:
        Tuple[np.ndarray]: A tuple containing the probabilities associated with each text description.

    Raises:
        Exception: If an error occurs while processing the image or performing classification.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()
    return probs

# test_function_code --------------------

def test_classify_image_with_text_descriptions():
    print("Testing started.")
    image_url = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
    texts = ['组织24小时相关', '宠物', '汽车']

    # 测试用1
    print("Testing case [1/3] started.")
    probs = classify_image_with_text_descriptions(image_url, texts)
    assert probs.shape == (1, len(texts)), f"Test case [1/3] failed: Expected (1, {len(texts)}) shape, got {probs.shape}"
    assert probs.argmax() == 1, f"Test case [1/3] failed: Expected the highest probability for 'pet', which is index 1, got index {probs.argmax()}"

    # 测试用2
    # 物体交换文本
    print("Testing case [2/3] started.")
    texts_alt = ['汽车1', '汽车2', '汽车3']
    probs_alt = classify_image_with_text_descriptions(image_url, texts_alt)
    assert probs_alt.shape == (1, len(texts_alt)),f"Test case [2/3] failed: Expected (1, {len(texts_alt)}) shape, got {probs_alt.shape}"

    # 测试用3
    # 使用不合法的URL
    print("Testing case [3/3] started.")
    try:
        classify_image_with_text_descriptions('invalid_url', texts)
        assert False, "Test case [3/3] failed: Invalid URL should raise an error."
    except:
        assert True
    print("Testing finished.")

# 运行测试函数
test_classify_image_with_text_descriptions()

# call_test_function_line --------------------

test_classify_image_with_text_descriptions()