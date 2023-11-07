from f00673_visualize_spectrogram import *
def test_visualize_spectrogram():
	# Test case 1
	spectrogram = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	visualize_spectrogram(spectrogram)

	# Test case 2
	spectrogram = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
	visualize_spectrogram(spectrogram)

test_visualize_spectrogram()
