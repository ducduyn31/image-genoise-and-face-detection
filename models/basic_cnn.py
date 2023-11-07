from torch import nn


class BasicCNN(nn.Module):

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride=1,
            downsample=None
    ):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, stride=stride, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_chans, out_channels=out_chans, stride=stride, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x if not self.downsample else self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += shortcut
        x = self.relu(x)

        return x
