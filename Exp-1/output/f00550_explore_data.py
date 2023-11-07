from typing import *
from PIL import Image
import matplotlib.pyplot as plt

def explore_data(dataset):
    example = dataset['train'][0]
    image_id = example['image_id']
    image = example['image']
    width = example['width']
    height = example['height']
    objects = example['objects']

    print(f'Image ID: {image_id}')
    print(f'Image Size: {width} x {height}')
    print(f'Number of Objects: {len(objects['id'])}')

    plt.imshow(image)
    plt.axis('off')
    plt.show()
