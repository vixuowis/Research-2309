from f00180_generate_picture import *
import matplotlib.pyplot as plt
import numpy as np

def test_generate_picture():
    '''
    Test the generate_picture function
    '''
    # Test case 1
    fig = generate_picture()
    assert isinstance(fig, plt.Figure)

    # Test case 2
    # Add more test cases here

