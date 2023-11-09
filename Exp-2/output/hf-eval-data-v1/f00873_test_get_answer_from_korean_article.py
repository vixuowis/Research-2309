def test_get_answer_from_korean_article():
    question = '서울의 수도는 무엇인가?'
    context = '서울은 대한민국의 수도이다.'
    assert get_answer_from_korean_article(question, context) == '대한민국'

test_get_answer_from_korean_article()