'''A version of ResNet for smaller input sizes.'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, input_channels=3, projection_output_dim=256):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.input_channels = input_channels

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        fc_input_size = 512 * block.expansion 
        self.fc = nn.Linear(fc_input_size, projection_output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0: 
            return x
        out = F.relu(self.bn1(self.conv1(x)))
        if layer == 1:
            return out
        out = self.layer1(out)
        if layer == 2:
            return out
        out = self.layer2(out)
        if layer == 3:
            return out
        out = self.layer3(out)
        if layer == 4:
            return out
        out = self.layer4(out)
        if layer == 5:
            return out

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        
        return out


def ResNet18(input_channels, projection_output_dim):
    return ResNet(BasicBlock, [2,2,2,2], input_channels, projection_output_dim)

def ResNet34(input_channels, projection_output_dim):
    return ResNet(BasicBlock, [3,4,6,3], input_channels, projection_output_dim)

def ResNet50(input_channels, projection_output_dim):
    return ResNet(Bottleneck, [3,4,6,3], input_channels, projection_output_dim)

def ResNet101(input_channels, projection_output_dim):
    return ResNet(Bottleneck, [3,4,23,3], input_channels, projection_output_dim)

def ResNet152(input_channels, projection_output_dim):
    return ResNet(Bottleneck, [3,8,36,3], input_channels, projection_output_dim)

