from typing import *
import json

def generate_python_code(data):
    '''
    Generate python code based on the instruction and example code provided.

    Args:
        data (dict): The input data containing the instruction and example code.

    Returns:
        str: The generated python code.
    '''
    code = ''

    # Extract the necessary information from the input data
    instruction = data['instruction']
    code_example = data['code_example']

    # Generate the markdown code snippet
    code += '```py\n'
    code += f'> {instruction}\n\n'
    code += f'{code_example}\n'
    code += '```'

    return code
