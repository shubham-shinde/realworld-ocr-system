from .base_model import BaseTextDetector


def get_model(name):
    if name == 'base':
        return BaseTextDetector
