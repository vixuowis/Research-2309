from f00391_bundle_callbacks import *
def test_bundle_callbacks():
    def callback1():
        print('Callback 1 executed')
    def callback2():
        print('Callback 2 executed')
    def callback3():
        print('Callback 3 executed')
    bundled_callback = bundle_callbacks([callback1, callback2, callback3])
    bundled_callback()

test_bundle_callbacks()
