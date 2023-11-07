from f00811_xla_generate import *
input_ids = tokenizer(input_string, return_tensors='tf')['input_ids']
output = xla_generate(input_ids)
print(output)
