def test_extract_non_compete_clause():
    context = 'The data protection provisions set forth in this agreement shall be in effect for a period of 2 years after the termination of services. The non-compete clause states that the service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services.'
    question = 'What are the terms of the non-compete clause?'
    expected_answer = 'the service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services'
    assert 'prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services' in extract_non_compete_clause(context, question)

test_extract_non_compete_clause()