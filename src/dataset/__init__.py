from .mnt import ImageTextDatasetMNT
from .synth import ImageTextDatasetSynt


def get_dataset(config, input_size):
    dimention = (input_size[0], input_size[1])
    is_gray = input_size[2] == 1
    name = config['dataset']
    if name == 'mnt':
        return lambda x: ImageTextDatasetMNT(x, gray=is_gray, out_size=dimention)
    elif name == 'synth':
        return lambda x: ImageTextDatasetSynt(len(x), gray=is_gray, out_size=dimention)
    return lambda: print('errrrr')
