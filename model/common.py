import warnings

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = k + (d - 1) * (k - 1) if isinstance(k, int) else [k + (d - 1) * (kk - 1) for kk in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [kk // 2 for kk in k]
    return p


class ConvBNSiLU(nn.Module):
    """Convolutional layer with Batch Normalization and SiLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, groups=1,
                 act=True):
        """
        Initializes the ConvBNSiLU layer.
        """
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the ConvBNSiLU layer.
        """
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    """Bottleneck layer with optional expansion and depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, expansion=0.5, group=1, shortcut=True):
        """
        Initializes the BottleNeck layer.
        """
        super(BottleNeck, self).__init__()
        self.conv1 = ConvBNSiLU(in_channels, int(out_channels * expansion), kernel_size=1)
        self.conv2 = ConvBNSiLU(int(out_channels * expansion), out_channels, kernel_size=3, groups=group, stride=1)
        self.shortcut = shortcut
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass through the BottleNeck layer.
        """
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling layer."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        """
        Initializes the SPPF layer.
        """
        super(SPPF, self).__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNSiLU((out_channels // 2) * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """
        Forward pass through the SPPF layer.
        """
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x1 = self.maxpool(x)
            x2 = self.maxpool(x1)
            x3 = self.maxpool(x2)
            x4 = self.maxpool(x3)
            return self.conv2(torch.cat((x1, x2, x3, x4), 1))


class C3(nn.Module):
    """C3 module with optional shortcut connection."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, group=1):
        """
        Initializes the C3 module.
        """
        out_c = int(out_channels * expansion)  # hidden channels
        super(C3, self).__init__()
        self.cv1 = ConvBNSiLU(in_channels, out_c, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNSiLU(in_channels, out_c, kernel_size=1, stride=1, padding=0)
        self.cv3 = ConvBNSiLU(out_c * 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.Sequential(
            *(BottleNeck(out_c, out_c, shortcut=shortcut, expansion=1.0, group=group) for _ in range(n))
        )

    def forward(self, x):
        """
        Forward pass through the C3 module.
        """
        return self.cv3(torch.cat((self.bottlenecks(self.cv1(x)), self.cv2(x)), 1))


class Concat(nn.Module):
    """Concatenation layer for multiple inputs."""

    def __init__(self, dimension=1):
        """
        Initializes the Concat layer.
        """
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        """
        Forward pass through the Concat layer.
        """
        return torch.cat(x, self.dimension)


# class Dectect(nn.Module):
#     """Detection layer for object detection tasks."""
#
#     def __init__(self, in_channels, out_channels, num_classes, anchors):
#         """
#         Initializes the Dectect layer.
#         """
#         super(Dectect, self).__init__()
#         self.conv = ConvBNSiLU(in_channels, out_channels, kernel_size=1)
#         self.num_classes = num_classes
#         self.anchors = anchors
#
#     def forward(self, x):
#         """
#         Forward pass through the Dectect layer.
#         """
#         return self.conv(x)  # Output shape: (batch_size, out_channels, height, width)# model/common.py


def main():
    # Example usage of the classes
    x = torch.randn(1, 128, 160, 160)  # Example input tensor
    model = C3(in_channels=128, out_channels=128, n=3, shortcut=True)



if __name__ == "__main__":
    main()
