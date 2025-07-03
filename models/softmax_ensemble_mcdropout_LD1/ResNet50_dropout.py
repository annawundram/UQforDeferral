# adapted from https://medium.com/data-scientists-diary/building-resnet-50-101-and-152-models-from-scratch-in-pytorch-f1e84cbafa63
import torch.nn as nn
import torch

class BottleneckBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, expansion=4):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)
        return x

class ResNet_50_mcdropout(nn.Module):
    
    def __init__(self, image_channels, num_classes, dropout_rate):
        
        super(ResNet_50_mcdropout, self).__init__()
        self.num_classes = num_classes
        self.expansion = 4
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_layers = [3, 4, 6, 3]
        
        #resnet layers
        self.layer1 = self.__make_layer(64, self.num_layers[0], stride=1)
        self.layer2 = self.__make_layer(128, self.num_layers[1], stride=2)
        self.layer3 = self.__make_layer(256, self.num_layers[2], stride=2)
        self.layer4 = self.__make_layer(512, self.num_layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = PermanentDropout(dropout_prob=dropout_rate)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        
    def __make_layer(self, out_channels, num_channels, stride):
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        layers = [BottleneckBlock(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * self.expansion
        for _ in range(1, num_channels):
            layers.append(BottleneckBlock(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class PermanentDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(PermanentDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.dropout_prob == 0:
            return x
        else:
            # Apply dropout during both training and evaluation
            return nn.functional.dropout(x, p=self.dropout_prob, training=True)