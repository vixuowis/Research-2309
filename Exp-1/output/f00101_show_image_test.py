from f00101_show_image import *
def test_show_image():
    img = np.random.rand(3, 32, 32)
    show_image(img)
    plt.show()

test_show_image()
