def test_find_most_related_faq():
    '''
    This function tests the find_most_related_faq function.
    '''
    # Define a list of FAQ sentences and a customer query
    faq_sentences = ['What is your return policy?', 'How do I track my order?', 'Do you offer discounts?']
    query = 'Can I return my order?'
    # Call the find_most_related_faq function
    most_related_faq = find_most_related_faq(faq_sentences, query)
    # Assert that the most related FAQ is correct
    assert most_related_faq in faq_sentences, 'The most related FAQ should be in the list of FAQ sentences.'

test_find_most_related_faq()