import torch
import torch.nn as nn

"""
不用预定参数，chanel，h,c

"""


def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):
    def __init__(self,chanels=3, num_classes=1000,loss_fun=None):
        super(MobileNet, self).__init__()

        self.loss_fun=loss_fun
        if self.loss_fun == None:
            self.loss_fun = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Conv2d(chanels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x,label=None):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        if label == None:
            return {'logits': x}
        else:
            loss = self.loss_fun(x, label)
            return {'logits': x, 'loss': loss}


def main():
    # 定义模型
    # 实例化网络
    net = MobileNet(num_classes=10)

    # 打印模型结构
    print(net)

    # 创建一个随机的输入张量
    input = torch.randn(1, 3, 224, 224)

    # 将输入传入模型，得到输出
    output = net(input)

    # 打印输出
    print(output['logits'].shape)

if __name__=='__main__':
    main()


