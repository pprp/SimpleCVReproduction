from pylatex import NoEscape,Package
from marktex import config

def get_config(c = None):
    if c is None:
        c = config
    return c

