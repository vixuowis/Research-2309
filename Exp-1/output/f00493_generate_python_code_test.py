from f00493_generate_python_code import *
def test_generate_python_code():
    code = generate_python_code()
    expected_code = '''
from transformers import Trainer

trainer = Trainer(model=model, args=args)

# Push the model to the Hub
trainer.push_to_hub()
'''
    assert code == expected_code, f'Expected code: {expected_code}
Generated code: {code}'
