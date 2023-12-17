# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_korean_question(question: str, context: str) -> str:
    """
    Provides answers to questions based on a given context in Korean language.

    Args:
        question (str): The question in Korean that needs to be answered.
        context (str): The context in Korean containing the information needed to answer the question.

    Returns:
        str: The answer extracted from the context for the given question.

    Raises:
        ValueError: If the question or context is empty.
    """
    if not question or not context:
        raise ValueError('The question and context should not be empty.')
    qa_pipeline = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = qa_pipeline(question=question, context=context)['answer']
    return answer

# test_function_code --------------------

def test_answer_korean_question():
    print("Testing started.")
    # Test cases with predefined questions and contexts
    questions_and_contexts = [
        ("엘렉트라 모델이 무엇입니까?", "엘렉트라(ELECTRA)는 트랜스포머 기반의 모델로서, 자연어 처리에 널리 사용됩니다."),
        ("코랏 데이터셋은 무엇을 위한 것입니까?", "코랏(KorQuAD) 데이터셋은 한국어 질의응답을 훈련하기 위한 것입니다.")
    ]
    # Test case 1
    print("Testing case [1/2] started.")
    question1, context1 = questions_and_contexts[0]
    answer1 = answer_korean_question(question1, context1)
    assert answer1 == "자연어 처리에 널리 사용되는 모델", f"Test case [1/2] failed: Expected '자연어 처리에 널리 사용되는 모델', got '{answer1}'"
    
    # Test case 2
    print("Testing case [2/2] started.")
    question2, context2 = questions_and_contexts[1]
    answer2 = answer_korean_question(question2, context2)
    assert answer2 == "한국어 질의응답 훈련", f"Test case [2/2] failed: Expected '한국어 질의응답 훈련', got '{answer2}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_korean_question()