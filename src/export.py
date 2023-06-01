import torch
import json
from torch import nn
import random
import cv2
import string as STR
from torch import optim
from tqdm import tqdm
from pathlib import Path
import wandb
import os
import sys
import traceback


def export_onnx(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    dataset = ImageTextDatasetMNT(Path('../datasets/detect_val_data'))
    dummy_input = dataset[0][0][None, ...]
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_input, "BaseTextDetector.onnx", verbose=True,
                      input_names=input_names, output_names=output_names)
    # onnx2tf -i BaseTextDetector.onnx -o torch_model


def export_pb(model_path='torch_model'):
    import tensorflow as tf  # noqa
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

    keras_model = tf.saved_model.load(model_path, tags=None, options=None)
    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(
        keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=model_path,
                      name='tf_model' + '.pb', as_text=False)
    # tensorflowjs_converter --input_format=tf_frozen_model --output_node_names="Func/PartitionedCall/output/_49" torch_model/tf_model.pb web_model/


if __name__ == '__main__':
    log = '--log' in sys.argv

    config = {
        'learning_rate': 0.0008,
        'epochs': 50,
        'train_size': 100*(10**4),
        'lr_step_size': 30,
        'lr_gamma': 0.8,
        'batch_size': 32
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # train(config, log, device)

    # export_onnx('BaseTextDetector.pt')
    export_pb()
