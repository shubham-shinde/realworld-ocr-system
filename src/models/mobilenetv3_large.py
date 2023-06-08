import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from .mobilenetv3_small import Block, PrintSize


class BreakPoint(nn.Module):
    def __init__(self):
        super(BreakPoint, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class MobileNetV3_Large(nn.Module):
    input_size = (3, 48, 320)

    def __init__(self, num_classes, act=nn.Hardswish):
        super(MobileNetV3_Large, self).__init__()
        # input = 224 * 224 * 3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=16,
                  out_size=16, act=nn.ReLU, se=False, stride=1),
            PrintSize('b1'),
            Block(kernel_size=3, in_size=16, expand_size=64,
                  out_size=24, act=nn.ReLU, se=False, stride=(2, 1)),
            PrintSize('b2'),
            Block(kernel_size=3, in_size=24, expand_size=72,
                  out_size=24, act=nn.ReLU, se=False, stride=1),
            PrintSize('b3'),
            Block(kernel_size=5, in_size=24, expand_size=72,
                  out_size=40, act=nn.ReLU, se=True, stride=(2, 1)),
            PrintSize('b4'),
            Block(kernel_size=5, in_size=40, expand_size=120,
                  out_size=40, act=nn.ReLU, se=True, stride=1),
            PrintSize('b5'),
            Block(kernel_size=5, in_size=40, expand_size=120,
                  out_size=40, act=nn.ReLU, se=True, stride=1),
            PrintSize('b6'),
            Block(kernel_size=3, in_size=40, expand_size=240,
                  out_size=80, act=act, se=False, stride=2),
            PrintSize('b7'),
            Block(kernel_size=3, in_size=80, expand_size=200,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize('b8'),
            Block(kernel_size=3, in_size=80, expand_size=184,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize('b9'),
            Block(kernel_size=3, in_size=80, expand_size=184,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize('b10'),
            Block(kernel_size=3, in_size=80, expand_size=480,
                  out_size=112, act=act, se=True, stride=1),
            PrintSize('b11'),
            Block(kernel_size=3, in_size=112, expand_size=672,
                  out_size=112, act=act, se=True, stride=1),
            PrintSize('b12'),
            Block(kernel_size=5, in_size=112, expand_size=672,
                  out_size=160, act=act, se=True, stride=(2, 1)),
            PrintSize('b13'),
            Block(kernel_size=5, in_size=160, expand_size=960,
                  out_size=160, act=act, se=True, stride=1),
            PrintSize('b14'),
            Block(kernel_size=5, in_size=160, expand_size=960,
                  out_size=160, act=act, se=True, stride=1),
            PrintSize('b15'),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.MaxPool2d(2)

        self.lstm = nn.LSTM(960, 64, bidirectional=True,
                            num_layers=2, dropout=0.2)
        self.linear3 = nn.Linear(128, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, targets=None, lengths=None):
        bs, c, h, w = x.size()
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out)
        out = out.permute([0, 3, 1, 2])
        out = out.view(bs, out.size(1), -1)
        out, _ = self.lstm(out)
        out = self.linear3(out)
        x = out.permute([1, 0, 2])

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


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MobileNetV3_Large(63).to(device)
    # b, c, h, w
    model_input = torch.rand([2, 3, 48, 320])
    model_output, loss = model(model_input)
    print('without target: ', model_output.shape, loss)
    model_input = torch.rand([2, 3, 48, 320])
    model_target = torch.randint(0, 64, (2, 20))
    model_output, loss = model(model_input, model_target)
    print('with target: ', model_output.shape, loss)
