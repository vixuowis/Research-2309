from f00827_image_classification import *
image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg'
result = image_classification(image_url)
print(*result, sep='\n')
# Expected Output:
# {'score': 0.4335, 'label': 'lynx, catamount'}
# {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
# {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
# {'score': 0.0239, 'label': 'Egyptian cat'}
# {'score': 0.0229, 'label': 'tiger cat'}

