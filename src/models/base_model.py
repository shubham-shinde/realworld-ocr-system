import torch
from torch import nn


class BaseTextDetector(nn.Module):
    input_size = (1, 32, 128)

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
                                       input_length, lengths)
            return x, loss
        return x, None


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = BaseTextDetector(64).to(device)
    model_input = torch.rand([1, 1, 32, 128])
    lengths = torch.tensor([6])
    model_output, loss = model(model_input, lengths=lengths)
    print('without target: ', model_output.shape, loss)
    model_input = torch.rand([1, 1, 32, 128])
    model_target = torch.randint(0, 64, (1, 20))
    lengths = torch.tensor([6])
    model_output, loss = model(model_input, model_target, lengths=lengths)
    print('with target: ', model_output.shape, loss)
