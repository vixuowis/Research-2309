# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def analyze_medical_case_report(text):
    """
    Analyze the medical case report to recognize biomedical entities.

    Args:
        text (str): The text of the physician's case report.

    Returns:
        dict: The recognized entities along with their labels.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('The text of the case report cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('d4data/biomedical-ner-all')
    model = AutoModelForTokenClassification.from_pretrained('d4data/biomedical-ner-all')
    ner_pipe = pipeline('token-classification', model=model, tokenizer=tokenizer)

    results = ner_pipe(text)
    entities = {result['entity']: result['word'] for result in results}
    return entities

# test_function_code --------------------

def test_analyze_medical_case_report():
    print("Testing started.")
    case_reports = [
        "The patient reported no recurrence of palpitations.",
        "A follow-up MRI of the brain with contrast showed no signs of tumor.",
        "The blood test results indicate an increase in white blood cell count."
    ]

    # Test case 1
    print("Testing case [1/3] started.")
    result_1 = analyze_medical_case_report(case_reports[0])
    assert isinstance(result_1, dict), "Test case [1/3] failed: The result should be a dictionary."

    # Test case 2
    print("Testing case [2/3] started.")
    result_2 = analyze_medical_case_report(case_reports[1])
    assert len(result_2) > 0, "Test case [2/3] failed: The result should contain recognized entities."

    # Test case 3
    print("Testing case [3/3] started.")
    assert isinstance(analyze_medical_case_report(case_reports[2]), dict), "Test case [3/3] failed: The result should be a dictionary."
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_medical_case_report()