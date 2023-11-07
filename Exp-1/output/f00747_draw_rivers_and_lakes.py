from typing import *
import matplotlib.pyplot as plt
import numpy as np

def draw_rivers_and_lakes():
    """Draws a picture of rivers and lakes."""
    plt.figure(figsize=(10, 10))

    # Draw rivers
    rivers = np.array([[1, 2], [3, 4], [5, 6]])
    plt.plot(rivers[:, 0], rivers[:, 1], 'b-', label='Rivers')

    # Draw lakes
    lakes = np.array([[7, 8], [9, 10], [11, 12]])
    plt.scatter(lakes[:, 0], lakes[:, 1], c='g', label='Lakes')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rivers and Lakes')
    plt.legend()

    # Show the plot
    plt.show()
