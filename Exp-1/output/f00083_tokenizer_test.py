from f00083_tokenizer import *
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]

encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='tf')
print(encoded_input)

# Output:
# {'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
# array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
#        [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
#        [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
#       dtype=int32)>,
#  'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
# array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#       dtype=int32)>,
#  'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
# array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
#       dtype=int32)>}
