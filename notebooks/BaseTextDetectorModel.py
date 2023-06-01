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


class ImageTextDatasetMNT(torch.utils.data.Dataset):

    letters = ' ' + STR.ascii_letters + STR.digits
    clasess = len(letters) + 1

    def __init__(self, dataset_path=Path('../datasets/mnt/'), size=None, only_synt=False):
        super(ImageTextDatasetMNT, self).__init__()
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
            dataset_path) for f in filenames if os.path.splitext(f)[1] in ['.jpg', '.jpeg', '.png']]
        if size is not None:
            result = result[:size]
        print('dataset count:', len(result))
        self.web2lowerset = json.load(open('words.json'))
        self.image_files = result
        self.only_synt = only_synt

    def generate_random_string(self):
        types = ['nu', 'wo', 'nuwo']
        ty = random.choice(types)
        string = ''
        if ty == 'nu':
            length = random.randint(1, 10)
            for i in range(length):
                string += str(random.randint(0, 10))
        if ty == 'wo':
            string = random.choice(self.web2lowerset)
        if ty == 'nuwo':
            t_string = list(random.choice(self.web2lowerset))
            indices = random.choices(
                list(range(len(t_string))), k=len(t_string)//3)
            for i in indices:
                t_string[i] = str(random.randint(0, 10))
            string = ''.join(t_string)
        if random.random() > 0.5:
            string = string.upper()

        # string = []
        # for i in range(29):
        #     string.append(random.choice(list(self.letters)))
        # string = ''.join(string)

        return string.replace('-', '')[:10]

    def generate_random_text_image(self):
        string = self.generate_random_string()
        img = torch.zeros((32, 128, 1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 24)
        fontScale = 0.6
        color = (255)
        thickness = random.choice(list(range(1, 2)))
        img = cv2.putText(img.numpy(), string, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return torch.tensor(img), string

    def __len__(self):
        return len(self.image_files)

    def pretransform(self, im, resized_sz=(32, 128)):
        sz = im.shape
        new_sz = resized_sz
        ratio = min(new_sz[0]/sz[0], new_sz[1]/sz[1])
        mid_sz = (int(sz[1]*ratio), int(sz[0]*ratio))  # (w, h)
        pd_sz = (new_sz[0]-mid_sz[1], new_sz[1]-mid_sz[0])
        mid_img = cv2.resize(im, mid_sz, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pd_sz[0]/2 - 0.1)
                          ), int(round(pd_sz[0]/2 + 0.1))
        left, right = int(round(pd_sz[1]/2 - 0.1)
                          ), int(round(pd_sz[1]/2 + 0.1))
        final = cv2.copyMakeBorder(
            mid_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return torch.tensor(final).to(torch.float32)

    def __getitem__(self, index):
        if self.only_synt == False:
            try:
                file = self.image_files[index]
                img = cv2.imread(file)
                string = file.split('/')[-1].split('.')[0].split('_')[1]
                img = self.pretransform(img).mean(-1)[..., None]
            except:
                traceback.print_exc()
                print('Error generating data:', file)
                img, string = self.generate_random_text_image()
        else:
            img, string = self.generate_random_text_image()

        img = img/255.0
        img = img.permute([2, 0, 1])

        string = list(string.replace('-', '').replace('.', ''))
        label = []
        max_length = 12
        for i in range(max_length):
            label.append(
                self.letters.index(string[i]) if len(string) > i else -1
            )
        # label = nn.functional.one_hot(
        #     torch.tensor(label), 63).float()
        label = torch.tensor(label)
        label += 1
        return img.to(torch.float32), label, max_length


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, ln: int):
        super(ImageTextDataset, self).__init__()
        self.web2lowerset = json.load(open('words.json'))
        self.letters = STR.ascii_letters + STR.digits
        self.ln = ln

    def generate_random_string(self):
        types = ['nu', 'wo', 'nuwo']
        ty = random.choice(types)
        string = ''
        if ty == 'nu':
            length = random.randint(1, 10)
            for i in range(length):
                string += str(random.randint(0, 10))
        if ty == 'wo':
            string = random.choice(self.web2lowerset)
        if ty == 'nuwo':
            t_string = list(random.choice(self.web2lowerset))
            indices = random.choices(
                list(range(len(t_string))), k=len(t_string)//3)
            for i in indices:
                t_string[i] = str(random.randint(0, 10))
            string = ''.join(t_string)
        if random.random() > 0.5:
            string = string.upper()

        # string = []
        # for i in range(29):
        #     string.append(random.choice(list(self.letters)))
        # string = ''.join(string)

        return string.replace('-', '')[:10]

    def generate_random_text_image(self):
        string = self.generate_random_string()
        img = torch.zeros((32, 128, 1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 24)
        fontScale = 0.6
        color = (255)
        thickness = random.choice(list(range(1, 2)))
        img = cv2.putText(img.numpy(), string, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return torch.tensor(img).transpose(2, 0).transpose(1, 2), string

    def __getitem__(self, _):
        img, string = self.generate_random_text_image()
        img = img/255
        string = list(string)
        label = []
        for i in range(10):
            label.append(
                self.letters.index(string[i]) + 1 if len(string) > i else 0
            )
        # label = nn.functional.one_hot(
        #     torch.tensor(label), 63).float()
        label = torch.tensor(label)
        label += 1
        return img, label, len(string)


class BaseTextDetector(nn.Module):
    def __init__(self, all_classes):
        super(BaseTextDetector, self).__init__()
        # input 32 * 128 * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 64, 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(64, 64, bidirectional=True,
                            num_layers=2, dropout=0.2)
        self.linear2 = nn.Linear(128, all_classes)

    def forward(self, x, targets=None, lengths=None):
        bs, c, w, h = x.size()
        # print(x.shape, 0)
        x = self.conv1(x)
        # print(x.shape, 1)
        x = self.conv2(x)
        # print(x.shape, 2)
        x = self.conv3(x)
        # print(x.shape, 3)
        x = self.conv4(x)
        # print(x.shape, 4)
        x = self.conv5(x)
        # print(x.shape, 5)
        x = self.conv6(x)
        # print(x.shape, 6)
        x = self.conv7(x)
        # print(x.shape, 7)
        x = x.permute([0, 3, 1, 2])
        # print(x.shape, 8)
        x = x.view(bs, x.size(1), -1)
        # print(x.shape, 9)
        x, _ = self.lstm(x)
        # print(x.shape, 10)
        x = self.linear2(x)
        x = x.permute([1, 0, 2])

        if targets is not None:
            log_softmax = nn.LogSoftmax(dim=2)(x)
            input_length = torch.full(
                size=(bs, ), fill_value=log_softmax.size(0), dtype=torch.int32)
            output_length = torch.full(
                size=(bs, ), fill_value=targets.size(1), dtype=torch.int32
            )
            # print(lengths)
            loss = nn.CTCLoss(blank=0)(log_softmax, targets,
                                       input_length, output_length)
            return x, loss
        return x, None


def model_text(data, max_needed=False):
    if max_needed is True:
        data = data.argmax(axis=1)
    letters = ImageTextDatasetMNT.letters
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


def train(config, log, device):
    model = BaseTextDetector(ImageTextDatasetMNT.clasess).to(device)

    wandb.init(
        # set the wandb project where this run will be logged
        project="text_detector_1",
        config=config,
        mode='disabled' if log == False else 'run'
    )

    # dataset
    dataset = ImageTextDatasetMNT(size=config['train_size'], only_synt=False)
    eval_dataset = ImageTextDatasetMNT(
        Path('../datasets/detect_val_data'), only_synt=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'])

    # model test
    # for img, label, lengths in eval_dataloader:
    #     out, loss = model(img, targets=label, lengths=lengths)
    #     print(out.shape, loss)

    # # train setting
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    # train loop
    lrs = []
    losses = []
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batches = 0
        print('epoch -', epoch)
        lrs.append(optimizer.param_groups[0]['lr'])
        print('learning rate', lrs[-1])
        pbar = tqdm(dataloader)
        for img, targets, lengths in pbar:
            targets = targets.to(device)
            img = img.to(device)

            # data check
            # temp_img = img[0].permute([1, 2, 0])
            # cv2.imshow(model_text(targets[0]), temp_img.cpu().detach().numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            lengths = lengths.to(device)
            batches += 1
            optimizer.zero_grad()
            _, loss = model(img.to(device), targets, lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(
                {'loss': epoch_loss/batches})
        batch_loss = epoch_loss/batches
        losses.append(batch_loss)
        eval_acc = 0
        for data in eval_dataloader:
            img = data[0].to(device)
            out, _ = model(img)
            acc = 0
            words = 0
            for ot, tg in zip(out.cpu().transpose(0, 1), data[1]):
                tg = model_text(tg)
                ot = model_text(ot.squeeze(), max_needed=True)
                rot = model_out_to_text(ot)
                for i, w in enumerate(tg):
                    words += 1
                    acc += int(w == rot[i] if len(rot) > i else False)
                print(tg, ot, rot)
            eval_acc = acc/words
            break
        print('accuracy:', eval_acc)

        wandb.log({"eval_acc": eval_acc, "batch_loss": batch_loss})

        scheduler.step()
        print('epoch_loss', losses[-1])
    wandb.finish()
    torch.save(model, 'BaseTextDetector.pt')


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
