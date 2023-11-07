from f00314_generate_python_code import *
data = {
    'answers': {'text': ['Answer 1', 'Answer 2']},
    'selftext': 'Selftext',
    'title': 'Title'
}

expected_code = '# Title: Title\n# Selftext: Selftext\n# Answers:\n# Answer 1: Answer 1\n# Answer 2: Answer 2\n'

assert generate_python_code(data) == expected_code
