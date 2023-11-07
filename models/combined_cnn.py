from typing import List, Any

import pytorch_lightning as pl
import torch
from timm.layers import trunc_normal_
from torch import nn

from models.basic_cnn import BasicCNN


class CombinedCNN(pl.LightningModule):

    def __init__(
            self,
            in_chans: int = 64,
            layers: List[int] = [2, 2, 2, 2],
            stride: int = 1,
            block=BasicCNN,
    ):
        super(CombinedCNN, self).__init__()

        # Resnet 18 base
        self.in_chans = in_chans

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_chans, stride=2, kernel_size=7, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_chans)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        last_out_chans = self.in_chans
        for depth in layers:
            self.layers.append(self._build_layer(block, last_out_chans, depth, stride))
            last_out_chans *= 2

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        # CelebASpoof has 40 features relating to the attributes of the face, 11 spoof types,
        # 5 types of illuminations, 3 types of environment
        self.fc_face_attr = nn.Linear(last_out_chans * block.expansion, 40)
        self.fc_spoof_type = nn.Linear(last_out_chans * block.expansion, 11)
        self.fc_illu = nn.Linear(last_out_chans * block.expansion, 5)
        self.fc_env = nn.Linear(last_out_chans * block.expansion, 3)
        self.fc_spoof = nn.Linear(last_out_chans * block.expansion, 2)

        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
            else:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _build_basic_block(
            self,
            block,
            out_chans,
            blocks_cnt,
            stride=1,
    ):
        downsample = None
        if stride != 1 or self.in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chans, out_chans * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(self.in_chans, out_chans, stride, downsample))
        self.in_chans = out_chans * block.expansion
        for _ in range(1, blocks_cnt):
            layers.append(block(self.in_chans, out_chans))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x_face_attr = self.fc_face_attr(x)
        x_spoof_type = self.fc_spoof_type(x)
        x_illu = self.fc_illu(x)
        x_env = self.fc_env(x)
        x_spoof = self.fc_spoof(x)

        x_face_attr = self.sigmoid(x_face_attr)
        x_spoof_type = self.sigmoid(x_spoof_type)
        x_illu = self.sigmoid(x_illu)
        x_env = self.sigmoid(x_env)
        x_spoof = self.sigmoid(x_spoof)

        return x_spoof
        # return torch.concat((x_face_attr, x_spoof_type, x_illu, x_env, x_spoof), dim=1)
