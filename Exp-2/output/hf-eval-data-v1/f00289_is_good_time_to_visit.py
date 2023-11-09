from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

def is_good_time_to_visit(image_path):
    '''
    This function takes the path of an image as input and returns whether it is a good time to visit the site in the image or not.
    It uses the pretrained ChineseCLIPModel for image classification.
    '''
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    image = Image.open(image_path)
    texts = ["好的参观时间", "不是好的参观时间"]

    inputs = processor(images=image, text=texts, return_tensors='pt')
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    result = dict(zip(texts, probs[0]))

    return result['好的参观时间'] > result['不是好的参观时间']