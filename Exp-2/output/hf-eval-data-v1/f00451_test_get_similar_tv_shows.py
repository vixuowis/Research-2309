def test_get_similar_tv_shows():
    '''
    This function tests the get_similar_tv_shows function.
    It uses a small set of TV show descriptions and checks if the output is a similarity matrix.
    '''
    tv_show_descriptions = [
      'A group of friends navigate through life in New York City.',
      'A detective solves crimes in the city of London.',
      'A group of teenagers discover they have superpowers.'
    ]
    similarity_matrix = get_similar_tv_shows(tv_show_descriptions)

    assert similarity_matrix.shape == (len(tv_show_descriptions), len(tv_show_descriptions)), 'The output shape is incorrect.'
    assert (similarity_matrix >= -1).all() and (similarity_matrix <= 1).all(), 'The similarity scores should be between -1 and 1.'

test_get_similar_tv_shows()