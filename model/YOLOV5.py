from model.common import ConvBNSiLU, C3, SPPF

import torch
import torch.nn as nn
from config.ConfigModelStructure import ConfigModelStructure


class YOLOV5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(YOLOV5, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.config = ConfigModelStructure()
        self._build_model()

    def _build_model(self):
        in_channels = self.in_channels
        for backbone in self.config.backbone:
            if backbone[2] == 'Conv':
                out_channels = backbone[3][0]
                kernel_size = backbone[3][1]
                stride = backbone[3][2]
                padding = backbone[3][3] if len(backbone[3]) > 3 else None
                self.layers.append(
                    ConvBNSiLU(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
                in_channels = out_channels
            elif backbone[2] == 'C3':
                out_channels = backbone[3][0]
                short_cut = backbone[3][1] if len(backbone[3]) > 1 else True
                self.layers.append(
                    C3(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        shortcut=short_cut,
                        n=backbone[1]
                    )
                )
                in_channels = out_channels
            elif backbone[2] == 'SPPF':
                out_channels = backbone[3][0]
                kernel_size = backbone[3][1]
                self.layers.append(
                    SPPF(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size
                         )
                )
        # for head in self.config.head:
        #     if head[2] == 'Conv':
        #         out_channels = head[3][0]
        #         kernel_size = head[3][1]
        #         stride = head[3][2]
        #         padding = head[3][3] if len(head[3]) > 3 else None
        #         self.layers.append(
        #             ConvBNSiLU(
        #                 in_channels=in_channels,
        #                 out_channels=out_channels,
        #                 kernel_size=kernel_size,
        #                 stride=stride,
        #                 padding=padding
        #             )
        #         )
        #         in_channels = out_channels
        #     elif head[2] == 'C3':
        #         out_channels = head[3][0]
        #         short_cut = head[3][1] if len(head[3]) > 1 else True
        #         self.layers.append(
        #             C3(
        #                 in_channels=in_channels,
        #                 out_channels=out_channels,
        #                 shortcut=short_cut,
        #                 n=head[1]
        #             )
        #         )
        #         in_channels = out_channels
        #     elif head[2] == 'nn.Upsample':
        #         size = head[3][0]
        #
        #         scale_factor = head[3][0]
        #
        #         mode = head[3][1] if len(head[3]) > 1 else 'nearest'
        #         self.layers.append(
        #             nn.Upsample(scale_factor=scale_factor, mode=mode)
        #         )



def main():
    # Example usage of the YOLOV5 model
    x = torch.randn(1, 3, 640, 640)  # Example input tensor
    model = YOLOV5(in_channels=3, num_classes=1)
    print(model.layers[0])  # Print the first layer


if __name__ == "__main__":
    main()
