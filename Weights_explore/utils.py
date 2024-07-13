import os


def project_root():
    ''' Returns the filepath of the current python file.'''
    return os.path.dirname(os.path.abspath(__file__))
