import time
import sys

# Function to display a loading spinner
# This function will display a loading spinner on the console
# It's useful to indicate that the system is undergoing maintenance

def loading_spinner():
    while True:
        for cursor in '|/-\\':
            print(cursor, end='', flush=True)
            time.sleep(0.1)
            print('\r', end='', flush=True)