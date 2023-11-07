from f00378_DataCollatorForSeq2Seq import *
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Test case 1
inputs = [{'input_ids': [1, 2, 3]}, {'input_ids': [4, 5]}]
expected_output = {'input_ids': [[1, 2, 3, 0], [4, 5, 0, 0]], 'labels': [[1, 2, 3, -100], [4, 5, -100, -100]]}
output = data_collator(inputs)
assert output == expected_output

# Test case 2
inputs = [{'input_ids': [1, 2]}, {'input_ids': [3, 4, 5]}]
expected_output = {'input_ids': [[1, 2, 0], [3, 4, 5]], 'labels': [[1, 2, -100], [3, 4, 5]]}
output = data_collator(inputs)
assert output == expected_output

# Test case 3
inputs = [{'input_ids': [1]}, {'input_ids': [2]}, {'input_ids': [3, 4]}]
expected_output = {'input_ids': [[1, 0, 0], [2, 0, 0], [3, 4, 0]], 'labels': [[1, -100, -100], [2, -100, -100], [3, 4, -100]]}
output = data_collator(inputs)
assert output == expected_output
