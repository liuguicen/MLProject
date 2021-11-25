from torch import nn


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
        """
        super(MaskRCNNHeads, self).__init__()
        for i, channels in enumerate(layers, 1):
            self.add_module('mask_fcn' + str(i), nn.Conv2d(in_channels, layers[1], kernel_size=3, stride=1, padding=1))
            self.add_module('relu' + str(i), nn.ReLU(inplace=True))
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        """
        :param in_channels: 目前256
        :param dim_reduced: 目前256
        """
        # 就是一个反卷积，扩大两倍，让生成的mask更加精细然后一个卷积
        super(MaskRCNNPredictor, self).__init__()
        self.add_module("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
