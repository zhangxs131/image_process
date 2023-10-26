# show_data
from torchvision import transforms,datasets

from PIL import Image
import numpy as np

def show_img(data_path,size=28):
    transform = transforms.Compose(
        [transforms.Resize([size, size]),transforms.ToTensor() ,transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][1])

    # 获取训练集中的第一张图像及其标签
    for i in range(20):
        img, label = train_dataset[i]

        # 因为图像已经被归一化，所以我们需要先反归一化
        img = img * 0.3081 + 0.1307

        # 将PyTorch张量转换为PIL图像
        img = transforms.ToPILImage()(img)

        # 显示图像
        img.save("data/mnist/test_dir/{}_{}.jpg".format(i,int(label)))


show_img("data/mnist")