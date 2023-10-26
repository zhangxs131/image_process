import torch
import torch.nn as nn
"""
[bs,3,224,224]
"""

# 定义DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([x, out], 1)
        return out

# 定义DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定义TransitionLayer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)

# 定义DenseNet
class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate,chanels=3, num_classes=10,loss_fun=None):
        super(DenseNet, self).__init__()

        self.loss_fun = loss_fun

        if self.loss_fun == None:
            self.loss_fun = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(chanels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(DenseBlock(num_layers, self.in_channels, growth_rate))
            self.in_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.trans_layers.append(TransitionLayer(self.in_channels, self.in_channels // 2))
                self.in_channels = self.in_channels // 2

        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x,label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for dense_block, trans_layer in zip(self.dense_blocks, self.trans_layers):
            x = dense_block(x)
            x = trans_layer(x)
        x = self.dense_blocks[-1](x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if label == None:
            return {'logits': x}
        else:
            loss = self.loss_fun(x, label)
            return {'logits': x, 'loss': loss}



def densenet121(pretrained=False, **kwargs):
    model = DenseNet(block_config=[6, 12, 24, 16],growth_rate=32, **kwargs)
    return model

def densenet169(pretrained=False,**kwargs):
    model= DenseNet(block_config=[6, 12, 32, 32],growth_rate=32, **kwargs)
    return  model

def densenet201(pretrained=False,**kwargs):
    model = DenseNet(block_config=[6, 12, 48, 32], growth_rate=32, **kwargs)
    return model

def main():
    # 定义模型
    model = densenet121(num_classes=10)

    # 打印模型结构
    print(model)

    # 创建一个随机的输入张量
    input = torch.randn(1, 3, 224, 224)

    # 将输入传入模型，得到输出
    output = model(input)

    # 打印输出
    print(output['logits'].shape)

if __name__=='__main__':
    main()