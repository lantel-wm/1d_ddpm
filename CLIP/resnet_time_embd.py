import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from typing import Type

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TEBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(TEBlock, self).__init__()
        # Multiplicative factor for the subsequent conv1d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels*self.expansion)
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2048, out_channels * 2)
        )
        
    def forward(self, x: torch.Tensor, time_emb=None) -> torch.Tensor:
        identity = x
        
        if time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (1 + scale) + shift

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
    
class ResNetTE(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[TEBlock],
        diffusion_steps: int  = 1000
    ) -> None:
        super(ResNetTE, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv1d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv1d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            padding_mode='circular'
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.time_embedding = SinusoidalPosEmb(dim=self.in_channels)
        self.time_mlp = nn.Sequential(
            self.time_embedding,
            nn.Linear(self.in_channels, self.in_channels * 4) ,
            nn.GELU(),
            nn.Linear(self.in_channels * 4, self.in_channels * 4)
        )
        # self.fc = nn.Linear(512*self.expansion, num_classes)
        
    def _make_layer(
        self, 
        block: Type[TEBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.ModuleList:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor, time) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        time = torch.Tensor(time).to(x.device)
        t = self.time_mlp(time)
        
        for layer in self.layer1:
            x = layer(x, t)
        for layer in self.layer2:
            x = layer(x, t)
        for layer in self.layer3:
            x = layer(x, t)
        for layer in self.layer4:
            x = layer(x, t)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    
    
if __name__ == '__main__':
    tensor = torch.rand([1, 1, 960])
    model = ResNetTE(img_channels=1, num_layers=18, block=TEBlock)
    
    # # from torchkeras import summary
    # # summary(model, input_shape=(1, 960))
    
    
    output = model(tensor)
    print(output.shape)