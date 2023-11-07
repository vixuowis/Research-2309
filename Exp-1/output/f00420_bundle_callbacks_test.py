from f00420_bundle_callbacks import *
def test_bundle_callbacks():
    def callback1():
        print('Callback 1 called')

    def callback2():
        print('Callback 2 called')

    def callback3():
        print('Callback 3 called')

    callbacks = [callback1, callback2, callback3]
    bundled_callback = bundle_callbacks(callbacks)
    bundled_callback()

# Test the bundle_callbacks function
if __name__ == '__main__':
    test_bundle_callbacks()
