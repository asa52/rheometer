class Hello:
    def __init__(self, hell=1):
        print("init called with {}".format(hell))

    def __enter__(self, hell=None):
        print("eneteer called")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exit called')

with Hello() as hello:
    print("jhbd")