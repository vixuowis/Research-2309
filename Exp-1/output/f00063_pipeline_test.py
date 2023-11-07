from f00063_pipeline import *
def test_pipeline():
    classifier = pipeline(model='facebook/bart-large-mnli')
    result = classifier('I have a problem with my iphone that needs to be resolved asap!!',
                        candidate_labels=['urgent', 'not urgent', 'phone', 'tablet', 'computer'])
    assert result == {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!',
                     'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'],
                     'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

test_pipeline()
