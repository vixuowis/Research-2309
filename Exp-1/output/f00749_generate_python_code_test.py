from f00749_generate_python_code import *
def test_generate_python_code():
    assert generate_python_code("Show me a tree", True) == agent.run("Show me a tree", return_code=True)
    assert generate_python_code("", False) == agent.run("")
    assert generate_python_code("Generate Python code", True) == agent.run("Generate Python code", return_code=True)
    assert generate_python_code("", True) == agent.run("", return_code=True)
    assert generate_python_code("Generate code snippet", False) == agent.run("Generate code snippet")

test_generate_python_code()
