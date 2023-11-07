from f00842_create_webserver import *
def test_create_webserver():
    q = asyncio.Queue()
    # Add test cases
    assert create_webserver(q) == None

if __name__ == '__main__':
    test_create_webserver()
