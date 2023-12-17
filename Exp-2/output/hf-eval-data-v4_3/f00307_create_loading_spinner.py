# requirements_file --------------------

import subprocess

requirements = ["time"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import time

# function_code --------------------

def create_loading_spinner(duration):
    """
    Create a loading spinner on the console for a given duration.

    Args:
        duration (int): The duration in seconds for the loading spinner to appear.

    Returns:
        None

    Raises:
        ValueError: If duration is not an integer or less than 0.
    """
    if not isinstance(duration, int) or duration < 0:
        raise ValueError('Duration must be a non-negative integer')
    spinner = ['-', '\\', '|', '/']
    for _ in range(duration * 10):
        for symbol in spinner:
            print(f'\r{symbol} loading...', end='')
            time.sleep(0.1)
    print('\r', end='')

# test_function_code --------------------

def test_create_loading_spinner():
    print('Testing started.')

    # Test case 1: Normal operation
    print('Testing case [1/3] started.')
    try:
        create_loading_spinner(2)
        print('\nTest case [1/3] succeeded.')
    except Exception as e:
        print(f'Test case [1/3] failed: {str(e)}')
    
    # Test case 2: Invalid duration type
    print('Testing case [2/3] started.')
    try:
        create_loading_spinner('3')
    except ValueError as e:
        print('Test case [2/3] succeeded.')
    else:
        print('Test case [2/3] failed: No ValueError for string duration.')

    # Test case 3: Negative duration
    print('Testing case [3/3] started.')
    try:
        create_loading_spinner(-1)
    except ValueError as e:
        print('Test case [3/3] succeeded.')
    else:
        print('Test case [3/3] failed: No ValueError for negative duration.')

    print('Testing finished.')

# call_test_function_line --------------------

test_create_loading_spinner()