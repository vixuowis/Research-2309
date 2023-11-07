from f00284_generate_python_code import *
data = {
    'instruction': 'Help me generate python code.',
    'code_example': 'print("Hello, World!")'
}

code = generate_python_code(data)
print(code)
