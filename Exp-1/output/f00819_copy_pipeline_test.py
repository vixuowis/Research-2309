from f00819_copy_pipeline import *
def test_copy_pipeline():
    copy_pipeline()

    classifier = pipeline(model='{your_username}/test-dynamic-pipeline', trust_remote_code=True)

    assert classifier('Some text') == expected_result
    assert classifier('Another text') == expected_result
    assert classifier('Yet another text') == expected_result
    assert classifier('One more text') == expected_result
    assert classifier('Last text') == expected_result
