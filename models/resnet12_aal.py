import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register
from .cbam import CBAMBlock

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class BlockCBAM(nn.Module):

    def __init__(self, inplanes, planes, downsample, return_attn=False):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.cbam = CBAMBlock(planes)

        self.downsample = downsample

        self.return_attn = return_attn

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out, attn = self.cbam(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        if self.return_attn:
            return out, attn
        else:
            return out


class ResNet12AAL(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = BlockCBAM(self.inplanes, planes, downsample, return_attn=True)
        self.inplanes = planes
        return block

    def forward(self, x):
        x_ori = x
        x, attn_1 = self.layer1(x)
        x, attn_2 = self.layer2(x)
        x, attn_3 = self.layer3(x)
        x, attn_4 = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        attns = [attn_1, attn_2, attn_3, attn_4]
        attn =  torch.stack([F.interpolate(a, x_ori.shape[2:]) for a in attns]).mean(dim=0)

        adv_noise = torch.randn_like(x_ori)
        x = x_ori + 0.01*attn*adv_noise

        x, attn_1 = self.layer1(x)
        x, attn_2 = self.layer2(x)
        x, attn_3 = self.layer3(x)
        x, attn_4 = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)

        return x


@register('resnet12-aal')
def resnet12_aal():
    return ResNet12AAL([64, 128, 256, 512])


@register('resnet12-wide-aal')
def resnet12_wide_aal():
    return ResNet12AAL([64, 160, 320, 640])

