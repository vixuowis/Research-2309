from f00188_generate_python_code import *
code_prompt = """model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

# By default, the output will contain up to 20 tokens
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]"""

generated_code = generate_python_code(code_prompt)
print(generated_code)
