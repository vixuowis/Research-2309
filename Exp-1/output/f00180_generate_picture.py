from typing import *
import matplotlib.pyplot as plt
import numpy as np

def generate_picture():
    '''
    Generate a picture of rivers and lakes

    Returns:
        matplotlib.figure.Figure: The generated picture
    '''
    fig, ax = plt.subplots()

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot data
    ax.plot(x, y)

    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Rivers and Lakes')

    return fig
