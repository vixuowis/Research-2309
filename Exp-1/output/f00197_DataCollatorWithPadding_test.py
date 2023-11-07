from f00197_DataCollatorWithPadding import *
import torch

input_ids = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
attention_mask = [[1, 1, 1], [1, 1, 1, 0], [1, 1]]
token_type_ids = [[0, 0, 0], [0, 0, 0, 1], [0, 0]]

encoded_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

batch = data_collator(encoded_inputs)

expected_batch = {'input_ids': torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7], [8, 9, 0, 0]]), 'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]]), 'token_type_ids': torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])}

assert batch == expected_batch
