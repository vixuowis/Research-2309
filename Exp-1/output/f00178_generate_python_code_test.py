from f00178_generate_python_code import *
import json
import unittest

def test_generate_python_code(unittest.TestCase):
    def test_example(self):
        request = "Draw me the picture of a capybara swimming in the sea"
        expected_code = f"""py
        # Your Python code goes here
        """
        
        generated_code = generate_python_code(request)
        self.assertEqual(generated_code, expected_code)

if __name__ == '__main__':
    unittest.main()
