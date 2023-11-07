from f00633_pipe import *
def test_pipe():
    example = dataset[0]
    image = Image.open(example['image_id'])
    question = example['question']
    print(question)
    assert pipe(image, question, top_k=1) == [{'score': 0.5498199462890625, 'answer': 'down'}]

