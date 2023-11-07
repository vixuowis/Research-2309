from typing import *
import matplotlib.pyplot as plt

def plot_histogram(speaker_counts):
    """Plot a histogram to visualize the data distribution.

    Args:
        speaker_counts (dict): A dictionary containing the count of examples for each speaker.

    Returns:
        None
    """
    plt.figure()
    plt.hist(speaker_counts.values(), bins=20)
    plt.ylabel("Speakers")
    plt.xlabel("Examples")
    plt.show()
