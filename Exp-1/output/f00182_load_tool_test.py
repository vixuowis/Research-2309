from f00182_load_tool import *
def test_load_tool():
    tool = load_tool("text-to-speech")
    audio = tool("This is a text to speech tool")


def main():
    test_load_tool()


if __name__ == "__main__":
    main()
