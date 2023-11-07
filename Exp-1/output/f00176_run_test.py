from f00176_run import *
def test_run():
    agent = Agent.load("models/dialogue")
    message = "Draw me a picture of rivers and lakes."
    agent.run(message)
