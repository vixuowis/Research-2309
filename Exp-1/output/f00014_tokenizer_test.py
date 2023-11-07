from f00014_tokenizer import *
texts = ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."]

pt_batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(pt_batch)
