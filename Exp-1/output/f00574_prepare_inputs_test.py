from f00574_prepare_inputs import *
text_queries = ["hat", "book", "sunglasses", "camera"]
images = [im1, im2, im3, im4]

inputs = prepare_inputs(text_queries, images)
print(inputs)
