from f00625_preprocess_data import *
dataset = Dataset({'question': ['What is the capital of France?', 'Who wrote Harry Potter?'],
                     'question_type': ['factoid', 'yesno'],
                     'question_id': [1, 2],
                     'image_id': [1001, 1002],
                     'answer_type': ['entity', 'person'],
                     'label.ids': [[1, 2, 3], [4, 5]],
                     'label.weights': [0.2, 0.8]})

processed_dataset = preprocess_data(dataset)
print(processed_dataset)
# Output: {'label.ids': [[1, 2, 3], [4, 5]], 'label.weights': [0.2, 0.8]}

