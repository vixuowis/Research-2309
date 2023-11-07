from f00170_run import *
def test_run(self):
    agent = Agent()
    query = "Caption the following image"
    image = "path/to/image.jpg"
    result = agent.run(query, image=image)
    assert isinstance(result, str)
    assert len(result) > 0
