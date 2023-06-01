from .mnt import ImageTextDatasetMNT


def get_dataset(name):
    if name == 'mnt':
        return ImageTextDatasetMNT
