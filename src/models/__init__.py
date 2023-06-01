from .base_model import BaseTextDetector
from .model_v1 import ModelV1


def get_model(name):
    if name == 'base':
        return BaseTextDetector
    if name == 'v1':
        return ModelV1
