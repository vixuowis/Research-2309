from f00272_run_question_answering import *
def test_run_question_answering():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    model = "my_awesome_qa_model"
    expected_result = {'score': 0.2058267742395401, 'start': 10, 'end': 95, 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
    assert run_question_answering(question, context, model) == expected_result

test_run_question_answering()
