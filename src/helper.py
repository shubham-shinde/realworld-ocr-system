import torch
import sys


def model_text(data, letters, max_needed=False):
    if max_needed is True:
        data = data.argmax(axis=1)
    return ''.join([letters[i-1] if i > 0 else ['.'][i] for i in data.numpy()])


def model_out_to_text(string):
    s = []
    for i in range(0, len(string)):
        if i > 0 and string[i] == string[i-1]:
            continue
        elif string[i] == '.':
            continue
        else:
            s.append(string[i])

    return ''.join(s)


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
