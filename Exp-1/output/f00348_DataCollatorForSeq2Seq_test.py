from f00348_DataCollatorForSeq2Seq import *
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

example1 = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
example2 = {'input_ids': [4, 5, 6, 7], 'attention_mask': [1, 1, 1, 1]}
example3 = {'input_ids': [8, 9], 'attention_mask': [1, 1]}

batch = data_collator([example1, example2, example3])
print(batch)
# Output: {'input_ids': tensor([[1, 2, 3, 0],
#                              [4, 5, 6, 7],
#                              [8, 9, 0, 0]]),
#          'attention_mask': tensor([[1, 1, 1, 0],
#                                    [1, 1, 1, 1],
#                                    [1, 1, 0, 0]]),
#          'labels': tensor([[1, 2, 3, 0],
#                           [4, 5, 6, 7],
#                           [8, 9, 0, 0]]),
#          'decoder_input_ids': tensor([[1, 2, 3, 0],
#                                      [4, 5, 6, 7],
#                                      [8, 9, 0, 0]])}
