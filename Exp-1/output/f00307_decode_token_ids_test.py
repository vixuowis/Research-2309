from f00307_decode_token_ids import *
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
token_ids = [10, 20, 30, 40, 50]

decoded_text = decode_token_ids(tokenizer, token_ids)
print(decoded_text)
# Output: ['Somatic hypermutation allows the immune system to react to drugs with the ability to adapt to a different environmental situation. In other words, a system of 'hypermutation' can help the immune system to adapt to a different environmental situation or in some cases even a single life. In contrast, researchers at the University of Massachusetts-Boston have found that 'hypermutation' is much stronger in mice than in humans but can be found in humans, and that it's not completely unknown to the immune system. A study on how the immune system']
