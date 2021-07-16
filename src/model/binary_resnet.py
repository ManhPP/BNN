import torch
from torch import nn
import torch.nn.functional as F

from src.layer.binary_layer import BinaryConv2d, BinaryLinear
from src.layer.binary_ops import binary_connect


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binarization = binary_connect(stochastic=True)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BinaryConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                nn.Hardtanh(),
                binary_connect(stochastic=True)
            )

    def forward(self, x):
        out = self.binarization(F.relu(self.bn1(self.conv1(x)), inplace=True))
        out = self.binarization(F.relu(self.bn2(self.conv2(out)), inplace=True))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BinaryResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BinaryResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = BinaryLinear(512 * block.expansion, num_classes)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BinaryResNet18(BinaryResNet):
    def __init__(self):
        super(BinaryResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])


class BinaryResNet34(BinaryResNet):
    def __init__(self):
        super(BinaryResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])


class BinaryResNet50(BinaryResNet):
    def __init__(self):
        super(BinaryResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])


class BinaryResNet101(BinaryResNet):
    def __init__(self):
        super(BinaryResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])


class BinaryResNet152(BinaryResNet):
    def __init__(self):
        super(BinaryResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = BinaryResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
