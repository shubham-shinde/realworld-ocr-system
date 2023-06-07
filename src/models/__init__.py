from .base_model import BaseTextDetector
from .model_v1 import ModelV1
from .mobilenetv3_large import MobileNetV3_Large
from .mobilenetv3_small import MobileNetV3_Small


def get_model(name):
    if name == 'base':
        return BaseTextDetector
    elif name == 'v1':
        return ModelV1
    elif name == 'mv3_l':
        return MobileNetV3_Large
    elif name == 'mv3_s':
        return MobileNetV3_Small
    else:
        raise NotImplementedError
