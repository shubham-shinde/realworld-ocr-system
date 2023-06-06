import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        # print(x.shape)
        return x


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 6, inplace=True)/6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True)/6
        return out


class SeModule(nn.Module):  # Squeeze and Excite
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size//reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            PrintSize(),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            PrintSize(),
            nn.BatchNorm2d(expand_size),
            PrintSize(),
            nn.ReLU(inplace=True),
            PrintSize(),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            PrintSize(),
            nn.Hardsigmoid()
        )
        self.in_size = in_size
        self.expand_size = expand_size

    def forward(self, x):
        out = self.se(x)
        return x * out


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None

        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size,
                          kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                PrintSize(),
                nn.BatchNorm2d(in_size),
                PrintSize(),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                PrintSize(),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size,
                          kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                PrintSize(),
                nn.BatchNorm2d(out_size),
                PrintSize(),
            )

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Large(nn.Module):
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
            PrintSize(),
            Block(kernel_size=3, in_size=16, expand_size=64,
                  out_size=24, act=nn.ReLU, se=False, stride=2),
            PrintSize(),
            Block(kernel_size=3, in_size=24, expand_size=72,
                  out_size=24, act=nn.ReLU, se=False, stride=1),
            PrintSize(),
            Block(kernel_size=5, in_size=24, expand_size=72,
                  out_size=40, act=nn.ReLU, se=True, stride=2),
            PrintSize(),
            Block(kernel_size=5, in_size=40, expand_size=120,
                  out_size=40, act=nn.ReLU, se=True, stride=1),
            PrintSize(),
            Block(kernel_size=5, in_size=40, expand_size=120,
                  out_size=40, act=nn.ReLU, se=True, stride=1),
            PrintSize(),
            Block(kernel_size=3, in_size=40, expand_size=240,
                  out_size=80, act=act, se=False, stride=2),
            PrintSize(),
            Block(kernel_size=3, in_size=80, expand_size=200,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize(),
            Block(kernel_size=3, in_size=80, expand_size=184,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize(),
            Block(kernel_size=3, in_size=80, expand_size=184,
                  out_size=80, act=act, se=False, stride=1),
            PrintSize(),
            Block(kernel_size=3, in_size=80, expand_size=480,
                  out_size=112, act=act, se=True, stride=1),
            PrintSize(),
            Block(kernel_size=3, in_size=112, expand_size=672,
                  out_size=112, act=act, se=True, stride=1),
            PrintSize(),
            Block(kernel_size=5, in_size=112, expand_size=672,
                  out_size=160, act=act, se=True, stride=2),
            PrintSize(),
            Block(kernel_size=5, in_size=160, expand_size=672,
                  out_size=160, act=act, se=True, stride=1),
            PrintSize(),
            Block(kernel_size=5, in_size=160, expand_size=960,
                  out_size=160, act=act, se=True, stride=1),
            PrintSize(),
        )

        self.conv2 = nn.Conv2d(160, 480, kernel_size=1,
                               stride=(2, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(480)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 40))

        self.lstm = nn.LSTM(480, 64, bidirectional=True,
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
    model_input = torch.rand([2, 3, 32, 128])
    model_output, loss = model(model_input)
    print('without target: ', model_output.shape, loss)
    model_input = torch.rand([2, 3, 32, 128])
    model_target = torch.randint(0, 64, (2, 20))
    model_output, loss = model(model_input, model_target)
    print('with target: ', model_output.shape, loss)
