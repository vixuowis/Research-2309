from f00895_inference_example import *
def test_inference_example():
    table, queries, answers, aggregation_predictions_string = inference_example()
    
    assert table.shape[0] == 3
    assert table.shape[1] == 2
    assert len(queries) == 3
    assert len(answers) == 3
    assert len(aggregation_predictions_string) == 3
    
    assert queries[0] == "What is the name of the first actor?"
    assert queries[1] == "How many movies has George Clooney played in?"
    assert queries[2] == "What is the total number of movies?"
    
    assert answers[0] == "Brad Pitt"
    assert answers[1] == "COUNT > 69"
    assert answers[2] == "SUM > 87, 53, 69"
    
    assert aggregation_predictions_string[0] == "NONE"
    assert aggregation_predictions_string[1] == "COUNT"
    assert aggregation_predictions_string[2] == "SUM"
    
    print("All tests pass")
