import torch
import torchvision
import torchvision.transforms as transforms

# transforms.Compose将多个transform组合起来使用
transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
     # 即：Normalized_image=(image-mean)/std
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################################################

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

import torch.nn as nn
import torch.nn.functional as F

# 定义一个卷积神经网络 在这之前先 从神经网络章节 复制神经网络，并修改它为3通道的图片(在此之前它被定义为1通道)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 定义一个损失函数和优化器 让我们使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器。
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # np.transpose,调换维度的索引值。
    #plt.imshow()函数负责对图像进行处理，并显示其格式，
    # 而plt.show()则是将plt.imshow()处理后的函数显示出来。
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show4Imgs():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    # show images
    # 给定 4D mini-batch Tensor， 形状为 (B x C x H x W),或者一个a list of image，
    # 做成一个size为(B / nrow, nrow)的雪碧图。
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    return images

def train(epoch):
    for epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def visOutput():
    outputs = net(show4Imgs())
    _, predicted = torch.max(outputs, 1)

    # Python join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
    # %:类似format
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))


def visAcc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # torch.max(input, dim)返回dim维度上的最大值
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def visPerClassAcc():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def changeDev():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print("device:")
    print(device)


device = "cpu"
if __name__ == '__main__':
    changeDev()
    net.to(device)
    # visOutput()
    train(2)
    visAcc()
    visPerClassAcc()

