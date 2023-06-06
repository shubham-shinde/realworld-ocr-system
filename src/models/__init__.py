from .base_model import BaseTextDetector
from .model_v1 import ModelV1
from .mobilenetv3_large import MobileNetV3_Large


def get_model(name):
    if name == 'base':
        return BaseTextDetector
    if name == 'v1':
        return ModelV1
    if name == 'mv3_l':
        return MobileNetV3_Large
