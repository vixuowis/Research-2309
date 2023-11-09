def test_get_loan_eligibility():
    document = "Our company policy restricts the loan applicant's eligibility to the citizens of United States. The applicant needs to be 18 years old or above and their monthly salary should at least be $4,000. FetchTypeOfYear: 2019."
    question = "Can anyone with a monthly salary of $3,000 apply?"
    answer = get_loan_eligibility(document, question)
    assert isinstance(answer, str), "The function should return a string."
    assert answer.lower() in ['yes', 'no'], "The answer should be either 'yes' or 'no'."

test_get_loan_eligibility()