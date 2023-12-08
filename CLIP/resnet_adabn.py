import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type
from .adabn import AdaBN1d


class AdaBNBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        diffusion_steps: int = 300,
    ) -> None:
        super(AdaBNBlock, self).__init__()
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
        # self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn1 = AdaBN1d(out_channels, diffusion_steps)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        # self.bn2 = nn.BatchNorm1d(out_channels * self.expansion)
        self.bn2 = AdaBN1d(out_channels * self.expansion, diffusion_steps)
        
    # @profile   
    def forward(self, x, class_id: int):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, class_id)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, class_id)
        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, AdaBN1d):
                    identity = layer(identity, class_id)
                else:
                    identity = layer(identity)
            # identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
    
class MultiIuputSeq(nn.Module):
    def __init__(self, modules):
        super(MultiIuputSeq, self).__init__()
        self.modules = modules
        
    def forward(self, x, class_id: int):
        for layer in self.modules:
            x = layer(x, class_id)
        return x
        

    
class AdaBNResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[AdaBNBlock],
        diffusion_steps: int = 300,
    ) -> None:
        super(AdaBNResNet, self).__init__()
        self.diffusion_steps = diffusion_steps
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
        # self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.bn1 = AdaBN1d(self.in_channels, diffusion_steps)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        # self.fc = nn.Linear(512*self.expansion, num_classes)
        
    def _make_layer(
        self, 
        block: Type[AdaBNBlock],
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
            downsample = nn.ModuleList([
                nn.Conv1d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                # nn.BatchNorm1d(out_channels * self.expansion),
                AdaBN1d(out_channels * self.expansion, self.diffusion_steps)
            ])
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
        # return nn.Sequential(*layers)
        # return MultiIuputSeq(layers)
        return nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x, class_id)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layer1:
            x = layer(x, class_id)
        for layer in self.layer2:
            x = layer(x, class_id)
        for layer in self.layer3:
            x = layer(x, class_id)
        for layer in self.layer4:
            x = layer(x, class_id)
        
        # x = self.layer1(x, class_id)
        # x = self.layer2(x, class_id)
        # x = self.layer3(x, class_id)
        # x = self.layer4(x, class_id)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    
    
if __name__ == '__main__':
    tensor = torch.rand([1, 1, 960])
    model = AdaBNResNet(img_channels=1, num_layers=18, block=AdaBNBlock)
    
    # # from torchkeras import summary
    # # summary(model, input_shape=(1, 960))
    
    
    output = model(tensor)
    print(output.shape)