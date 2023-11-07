from f00688_generate_text import *
input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])
output = generate_text(input_ids)
print(output)
