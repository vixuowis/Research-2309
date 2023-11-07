from f00102_collate_fn import *
def test_collate_fn():
    # Test case 1
    batch = [{'pixel_values': [1, 2, 3], 'labels': [0, 1, 2]},
             {'pixel_values': [4, 5, 6], 'labels': [3, 4, 5]},
             {'pixel_values': [7, 8, 9], 'labels': [6, 7, 8]}]
    expected_output = {'pixel_values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                      'pixel_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      'labels': [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}
    assert collate_fn(batch) == expected_output
    
    # Test case 2
    batch = [{'pixel_values': [1, 2, 3], 'labels': [0, 1, 2]},
             {'pixel_values': [4, 5, 6], 'labels': [3, 4, 5]}]
    expected_output = {'pixel_values': [1, 2, 3, 4, 5, 6],
                      'pixel_mask': [1, 1, 1, 1, 1, 1],
                      'labels': [[0, 1, 2], [3, 4, 5]]}
    assert collate_fn(batch) == expected_output
    
    # Test case 3
    batch = [{'pixel_values': [1, 2, 3], 'labels': [0, 1, 2]},
             {'pixel_values': [4, 5, 6], 'labels': [3, 4, 5]},
             {'pixel_values': [7, 8, 9], 'labels': [6, 7, 8]},
             {'pixel_values': [10, 11, 12], 'labels': [9, 10, 11]}]
    expected_output = {'pixel_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      'pixel_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      'labels': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]}
    assert collate_fn(batch) == expected_output
    
    # Test case 4
    batch = [{'pixel_values': [], 'labels': []}]
    expected_output = {'pixel_values': [], 'pixel_mask': [], 'labels': [[]]}
    assert collate_fn(batch) == expected_output
    
    # Test case 5
    batch = []
    expected_output = {'pixel_values': [], 'pixel_mask': [], 'labels': []}
    assert collate_fn(batch) == expected_output
    
    print('All test cases pass')

test_collate_fn()
