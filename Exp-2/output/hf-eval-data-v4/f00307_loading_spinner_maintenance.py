# requirements_file --------------------

!pip install -U sys, time

# function_import --------------------

import sys
import time

# function_code --------------------

def loading_spinner_maintenance(timeout=5):
    # This function displays a simple loading spinner
    # for the specified timeout duration.
    spinner = ['-', '\\', '|', '/']
    start_time = time.time()
    while time.time() - start_time < timeout:
        for phase in spinner:
            sys.stdout.write('\r' + phase)
            sys.stdout.flush()
            time.sleep(0.2)

# test_function_code --------------------

def test_loading_spinner_maintenance():
    print('Testing loading_spinner_maintenance function...')

    # Store the original stdout object
    original_stdout = sys.stdout

    # Redirect stdout to capture the spinner phases
    sys.stdout = StringIO()

    # Perform the test
    loading_spinner_maintenance(timeout=1)
    output = sys.stdout.getvalue()

    # Reset the stdout to original
    sys.stdout = original_stdout

    # Check if the captured output contains expected spinner phases
    assert '-' in output and '\\' in output and '|' in output and '/' in output, 'Spinner phases are missing in the output'
    print('Test passed.')

# Run the test
if __name__ == '__main__':
    test_loading_spinner_maintenance()