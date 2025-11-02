import sys, os
from functools import wraps

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# decorater used to block function printing to the console
def blockPrinting(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Save the current state of sys.stdout
        original_stdout = sys.stdout
        # Redirect sys.stdout to os.devnull
        sys.stdout = open(os.devnull, 'w')
        try:
            # Call the function with printing blocked
            return func(*args, **kwargs)
        finally:
            # Restore the original sys.stdout
            sys.stdout.close()
            sys.stdout = original_stdout

    return func_wrapper