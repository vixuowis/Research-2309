def test_loading_spinner():
    try:
        loading_spinner()
    except KeyboardInterrupt:
        print('Loading spinner function works correctly')
    except Exception as e:
        print('Loading spinner function failed:', e)

test_loading_spinner()