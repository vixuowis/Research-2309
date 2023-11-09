def test_rank_search_results():
    """
    This function tests the 'rank_search_results' function.
    It uses a sample query and passages, and checks if the function returns the expected output.
    """
    query = "Example search query"
    passages = [
        "passage 1",
        "passage 2",
        "passage 3"
    ]
    expected_output = [('passage 1', 0.9), ('passage 2', 0.8), ('passage 3', 0.7)]  # assuming these are the expected scores

    output = rank_search_results(query, passages)
    for i in range(len(output)):
        assert output[i][0] == expected_output[i][0]  # check if the passages are correctly ranked
        assert abs(output[i][1] - expected_output[i][1]) < 0.1  # check if the scores are approximately correct

test_rank_search_results()