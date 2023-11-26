# 背景
## 文章介绍内容
![[Pasted image 20231102145245.png]]
- AlexNet。它是第一个在大规模视觉竞赛中击败传统计算机视觉模型的大型神经网络；
- VGG（使用重复块的网络）。它利用许多重复的神经网络块；
- NiN（网络中的网络）。它重复使用由卷积层和$1\times 1$卷积层（用来代替全连接层）来构建深层网络;
- GoogLeNet（含并行连结的网络）。它使用并行连结的网络，通过不同窗口大小的卷积层和最大汇聚层来并行抽取信息；
- ResNet（残差网络）。它通过残差块构建跨层的数据通道，是计算机视觉中最流行的体系架构；
- DenseNet（稠密连接网络）。它的计算成本很高，但给我们带来了更好的效果。

## 先前机器学习——核方法
获取数据——>根据光学、几何学预处理——>特征提取算法（最重要）——>喜欢的分类器（or核方法）
所以早期的机器学习最重要的是特征提取。所以我先前学习的机器视觉主要讲述的就是上述流程。所以那门课被命名为机器视觉而不是计算机视觉，这有本质差别。

# AlexNet
2012年出现ALexNet。
![[Pasted image 20231102123324.png]]


AlexNet和LeNet区别：

1. 深度和规模：
   - AlexNet：AlexNet是一个**相对较大和深的神经网络**，它在2012年的ImageNet大规模视觉识别挑战赛上获胜。它有8个卷积层和3个全连接层，拥有60百万个参数。
   - LeNet：LeNet是一个相对较小和浅的神经网络，它是早期的卷积神经网络，由Yann LeCun在1998年创建。LeNet具有2个卷积层和3个全连接层，参数数量较少。

2. 激活函数：
   - AlexNet：AlexNet使用的主要激活函数是**ReLU**（Rectified Linear Unit），这对于训练深度神经网络非常有帮助，因为它有助于解决梯度消失问题。
   - LeNet：LeNet使用的激活函数是S型函数（**Sigmoid**）和双曲正切函数（tanh），这些函数在训练深度网络时可能会导致梯度消失问题。

3. 数据集：
   - AlexNet：AlexNet主要是为了解决ImageNet数据集上的大规模**图像分类问题**而设计的。
   - LeNet：LeNet最初是为**手写数字识别任务**设计的，用于MNIST数据集。

4. 架构：
   - AlexNet：AlexNet引入了重要的深度学习概念，如卷积层堆叠、局部响应归一化（LRN）和丢弃（**Dropout**）等。它具有更大的卷积核和更深的网络结构。
   - LeNet：LeNet是早期的卷积神经网络，较小且较浅，主要由卷积层、池化层和全连接层构成。


## 网络框架
```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
```

## 应用
```python
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


# VGG
VGG是Oxford的**V**isual **G**eometry **G**roup的组在2014年提出的，提出使用块的想法。
[VGG-16可视化](https://dgschwend.github.io/netscope/#/preset/vgg-16)
![[Pasted image 20231102125305.png]]
## VGG块
经典卷积神经网络的基本组成部分是下面的这个序列：
1. 带填充以保持分辨率的卷积层；
1. 非线性激活函数，如ReLU；
1. 汇聚层，如最大汇聚层。

所以作者想要VGG块里也基本是这个架构：
1. $3\times3$卷积核、填充为1（保持高度和宽度）的卷积层
2. ReLU激活函数
3. $2 \times 2$汇聚窗口、步幅为2（每个块后的分辨率减半）的最大汇聚层。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 定义一个vgg_block函数，用于创建VGG网络中的卷积块
# num_convs: 卷积层的数量
# in_channels: 输入通道数
# out_channels: 输出通道数
def vgg_block(num_convs, in_channels, out_channels):
    layers = []  # 创建一个用于存储层的列表
    for _ in range(num_convs):
        # 添加卷积层，其中kernel_size=3表示3x3的卷积核，padding=1表示使用1像素的填充
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # 添加ReLU激活函数，用于引入非线性
        layers.append(nn.ReLU())
        in_channels = out_channels  # 更新输入通道数以供下一层使用
    # 添加最大池化层，kernel_size=2表示2x2的最大池化窗口，stride=2表示步幅为2
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # 返回一个包含所有卷积和池化层的Sequential容器
    return nn.Sequential(*layers)

```

## VGG-11
有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。
第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。
由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。

```python
# 定义VGG网络的卷积层结构
#使用元组(conv_arch)表示每个卷积块的卷积层数量和输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 定义一个函数vgg，用于创建VGG网络
def vgg(conv_arch):
    conv_blks = []  # 创建一个列表，用于存储VGG网络的卷积块
    in_channels = 1  # 输入图像的通道数

    # 遍历conv_arch中的元组，每个元组包含(num_convs, out_channels)参数
    # num_convs表示卷积层的数量，out_channels表示输出通道数
    for (num_convs, out_channels) in conv_arch:
        # 使用vgg_block函数创建卷积块，并将其添加到conv_blks列表中
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels  # 更新输入通道数以供下一个卷积块使用

    # 创建一个Sequential容器，将卷积块、展平层、全连接层和输出层按顺序组合成整个网络
    return nn.Sequential(
        *conv_blks,  # 将卷积块添加到Sequential容器中
        nn.Flatten(),  # 将卷积层输出展平
        nn.Linear(out_channels * 7 * 7, 4096),  # 第一个全连接层
        nn.ReLU(),  # ReLU激活函数
        nn.Dropout(0.5),  # 丢弃层，用于防止过拟合
        nn.Linear(4096, 4096),  # 第二个全连接层
        nn.ReLU(),  # ReLU激活函数
        nn.Dropout(0.5),  # 丢弃层
        nn.Linear(4096, 10)  # 输出层，这里假设有10个类别
    )

# 创建VGG网络实例，使用conv_arch参数指定的卷积层结构
net = vgg(conv_arch)

```

VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）


AlexNet和VGG关键的区别：

1. **深度和参数数量**：
   - AlexNet：AlexNet是较早的卷积神经网络架构，包含8个卷积层和3个全连接层，总参数数量约为60-65百万。虽然在当时非常深，但相对于后来的架构来说，它相对较浅。
   - VGG：VGG网络的深度较大，有两个主要版本，分别包括16个和19个卷积层，其中每个卷积层都使用3x3的卷积核。这使得VGG的总参数数量明显更多，通常在138-144百万之间，使其成为更深的架构。

2. **卷积核大小**：
   - AlexNet：AlexNet在较早的时候引入了较大的卷积核，其中一些卷积层使用了11x11和5x5的卷积核。
   - VGG：VGG标志着更小的卷积核的回归，**大多数卷积层都使用3x3的卷积核**，这提供了更好的特征提取和参数共享。3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为 3x(9xC^2) ，如果直接使用7x7卷积核，其参数总量为 49xC^2 ，这里 C 指的是输入和输出的通道数。很明显，27xC^2小于49xC^2，即减少了参数；而且3x3卷积核有利于更好地保持图像性质。

3. **池化**：
   - AlexNet：AlexNet使用了局部响应归一化（LRN）来增强泛化能力，但在后来的架构中被较少使用。
   - VGG：VGG使用了**最大池化层**，这成为后续卷积神经网络架构的标准做法。（最大池化+卷积~=[平移不变性](https://zhangting2020.github.io/2018/05/30/Transform-Invariance/))

4. **网络结构**：
   - AlexNet：AlexNet的架构相对复杂，包括多个并行的卷积分支和一些特殊技巧，如丢弃（Dropout）。
   - VGG：VGG的结构相对简单和一致，由一系列连续的卷积层和池化层组成，这种一致性有助于简化网络设计。



## 应用
由于VGG-11比AlexNet计算量更大，因此构建了一个通道数较少的网络,训练过程和AlexNet相似
```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


## Q&A
> 与AlexNet相比，VGG的计算要慢得多，而且它还需要更多的显存。分析出现这种情况的原因。

VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG有3个全连接层。

> 请参考VGG论文 :`Simonyan.Zisserman.2014`中的表1构建其他常见模型，如VGG-16或VGG-19。

```python
def vgg16():
    conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    return vgg(conv_arch)

def vgg19():
    conv_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
    return vgg(conv_arch)

```


# NiN
VGG、AlexNet、LeNet统一模式：卷积汇聚提取空间结构特征+全连接处理特征的表征。
AlexNet和VGG主要改进就是扩大和加深这两个模块。
但是我们知道之前VGG的全连接层：1.破坏表征的空间结构 2.参数量巨大。
NiN的设计旨在增强深度卷积神经网络对特征的表达能力，并减少参数数量。
[NiN可视化](https://cwlacewe.github.io/netscope/#/preset/nin)


![[Pasted image 20231102133203.png]]

## NiN块
NIN（Network In Network）是一种卷积神经网络结构，由Min Lin等人于2014年提出。
卷积层的输入和输出由四维张量组成：样本、通道、高度和宽度。
全连接层的输入和输出：样本和特征的二维张量。
NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层。即视为$1\times 1$卷积层。
从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。

关键特点和概念：

1. **全局平均池化（Global Average Pooling）**：NiN引入了全局平均池化层，这是一种不同于传统池化的技术。全局平均池化将每个特征图降维为一个标量，使得网络更加紧凑。

2. **1x1卷积核**：NiN网络使用1x1的卷积核，这被称为“网络内卷积”（Network in Network）。

3. **多分支结构**：NiN采用了多分支的结构，其中每个分支使用不同的1x1和3x3卷积核来提取特征。这些分支的特征图被级联起来，从而增加了特征的多样性和丰富性。

4. **去除全连接层**：与传统的深度卷积神经网络不同，NiN去除了全连接层，这减少了网络的参数数量，同时减轻了过拟合的风险。全局平均池化层取代了全连接层，将特征图映射为类别概率。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 定义一个NiN块（Network in Network块）
# 输入参数：
# - in_channels: 输入通道数
# - out_channels: 输出通道数
# - kernel_size: 卷积核大小
# - strides: 卷积步幅
# - padding: 卷积填充
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    # 使用Sequential容器定义NiN块，包含以下操作：
    # 1. 一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为kernel_size，步幅为strides，填充为padding
    # 2. ReLU激活函数，用于引入非线性
    # 3. 一个1x1的卷积层，用于增加网络的非线性特征
    # 4. ReLU激活函数
    # 5. 另一个1x1的卷积层，用于增加网络的非线性特征
    # 6. ReLU激活函数
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1卷积
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1卷积
        nn.ReLU())

```

## NiN模型

```python
# 构建NiN网络
net = nn.Sequential(
    # 第一个NiN块，输入通道数为1，输出通道数为96，使用11x11的卷积核，步幅为4，填充为0
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # 最大池化层，窗口大小为3x3，步幅为2
    nn.MaxPool2d(3, stride=2),
    # 第二个NiN块，输入通道数为96，输出通道数为256，使用5x5的卷积核，步幅为1，填充为2
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    # 最大池化层，窗口大小为3x3，步幅为2
    nn.MaxPool2d(3, stride=2),
    # 第三个NiN块，输入通道数为256，输出通道数为384，使用3x3的卷积核，步幅为1，填充为1
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    # 最大池化层，窗口大小为3x3，步幅为2
    nn.MaxPool2d(3, stride=2),
    # 丢弃层，用于减少过拟合风险，丢弃率为0.5
    nn.Dropout(0.5),
    # 最后一个NiN块，将输出通道数降至类别数10，使用3x3的卷积核，步幅为1，填充为1
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 全局平均池化层，将特征图降维至1x1的大小
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转换成二维的输出，形状为(批量大小, 10)
    nn.Flatten()
)

```

## 应用

和之前相似。
```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


## Q&A
>为什么NiN块中有两个$1\times 1$卷积层？

目的：好的抽象特征需要对输入数据做高度的非线性变换。（深度学习的多层非线 性结构使其具备强大的特征表达能力和对复杂任务的建模能力。）（就类比于SVM的超平面，能够分类更强）

以往的CNN的做法：
1. 使用多输出通道即多卷积核，去覆盖同一特征野（input data patch）的所有variations。
2. 多个CNN层的stacking来获取前层特征的更高抽象，但是参数and计算量up。


现在的NiN的做法。

第一个$1\times 1$的卷积层：对前一层所有的feature map的线性组合。（对于前一层各个channel 对应其自身的一个卷积核，每个卷积核就是为了提取自己的特征，所以每个channel对应自己的feature map）。再进行Relu的非线性变换，就实现了特征的整合和非线性抽象。（和传统CNN没啥区别）。

第二个$1\times 1$的卷积层：对输入数据进行高度的非线性变换。在不增加参数和计算量的同时实现对特征的高度整合和抽象。

简单来说：第一个1x1卷积层实现featuremap的提取，第二个1x1卷积层进行featuremap的组合

>为什么最后一层用global average pooling代替全连接层

![[Pasted image 20231102144101.png]]

Global Average Pooling的优点如下：

1. 不引入新的参数，避免了全连接层带来的参数数量增加和过度导入；
2. 增加网络的可解释性，输出的每个通道对应一个类别；
3. 通过实验发现，全局均值池化还有正则化的作用。

区别GMP和GAP。
GMP只取每个feature map中的最重要的region，这样会导致，一个feature map中哪怕只有一个region是和某个类相关的，这个feature map都会对最终的预测产生很大的影响。而GAP则是每个region都进行了考虑，这样可以保证不会被一两个很特殊的region干扰。这篇论文有更详细的说明。

# GoogLeNet

[GoogLeNet可视化](https://cwlacewe.github.io/netscope/#/preset/googlenet)

创新：解决了什么样大小的卷积核最合适的问题.
毕竟，以前流行的网络使用小到$1 \times 1$，大到$11 \times 11$的卷积核。本文的一个观点是，有时使用不同大小的卷积核组合是有利的。

## Inception block

![[Pasted image 20231102144924.png]]


前三条路径使用窗口大小为$1\times 1$、$3\times 3$和$5\times 5$的卷积层，从不同空间大小中提取信息。
中间的两条路径在输入上执行$1\times 1$卷积，以减少通道数，从而降低模型的复杂性。
第四条路径使用$3\times 3$最大汇聚层，然后使用$1\times 1$卷积层来改变通道数。
这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的**输出在通道维度上连结**，并构成Inception块的输出。在Inception块中，通常调整的**超参数是每层输出通道数**。


```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```



每个Inception模块包括多个不同大小的卷积核和池化操作，以便在不同尺度上提取特征。（比如不同大小的滤波器可以有效地识别不同范围的图像细节）。这种设计增加了网络的表达能力，同时减少了参数数量。

## GoogLeNet

![[Pasted image 20231102151223.png]]


```python
# 定义网络的不同部分（模块）
# 第一部分：b1
#64个通道、$7\times 7$卷积层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 7x7的卷积层，输入通道数1，输出通道数64，步幅2，填充3
    nn.ReLU(),  # ReLU激活函数
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，窗口大小3x3，步幅2，填充1
)

# 第二部分：b2
#两个卷积层：第一个卷积层是64个通道、$1\times 1$卷积层；第二个卷积层使用将通道数量增加三倍的$3\times 3$卷积层。
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),  # 1x1的卷积层，用于降维通道数，输入通道数64，输出通道数64
    nn.ReLU(),  # ReLU激活函数
    nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 3x3的卷积层，输入通道数64，输出通道数192，填充1
    nn.ReLU(),  # ReLU激活函数
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，窗口大小3x3，步幅2，填充1
)

# 第三部分：b3
#计算输出通道数：
#第一个Inception块的输出通道数为64+128+32+32=256。

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),  # 使用Inception模块
    Inception(256, 128, (128, 192), (32, 96), 64),  # 使用Inception模块
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，窗口大小3x3，步幅2，填充1
)

# 第四部分：b4
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),  # 使用Inception模块
    Inception(512, 160, (112, 224), (24, 64), 64),  # 使用Inception模块
    Inception(512, 128, (128, 256), (24, 64), 64),  # 使用Inception模块
    Inception(512, 112, (144, 288), (32, 64), 64),  # 使用Inception模块
    Inception(528, 256, (160, 320), (32, 128), 128),  # 使用Inception模块
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，窗口大小3x3，步幅2，填充1
)

# 第五部分：b5
#需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1。


b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),  # 使用Inception模块
    Inception(832, 384, (192, 384), (48, 128), 128),  # 使用Inception模块
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化层，输出大小固定为1x1
    nn.Flatten()  # 将输出拉平为一维向量
)

# 定义整体的网络结构
#最后接上一个输出个数为标签类别数的全连接层。
net = nn.Sequential(
    b1, b2, b3, b4, b5,  # 依次堆叠不同部分
    nn.Linear(1024, 10)  # 全连接层，用于输出最终的类别预测结果
)

```




# 批量规范化 (batch normalization)

背景：
深度学习的挑战：在深度学习早期，研究人员发现训练非常深的神经网络是困难的，因为随着网络层数的增加，梯度消失和梯度爆炸问题变得更加明显。这导致了深层网络的性能不如较浅的网络，限制了深度学习模型的应用范围。

前期的解决方法：为了解决深度网络的训练问题，一些方法被提出，包括使用更复杂的激活函数、批归一化（Batch Normalization）等。这些方法有助于缓解梯度问题，但仍然难以训练非常深的网络。

趋势：研究者逐渐认识到，深度网络的问题并非深度本身导致的，而是由于网络层之间的映射不容易优化。因此，需要一种新的网络结构来解决这个问题。（这要到之后的ResNet了）

我们先提出这个批归一化。

![[Pasted image 20231102180419.png]]

在深度学习中，每一层的输入数据分布也在不断变化。模型的学习过程就是要使每一层适应这些不断变化的输入数据分布。如果不使用批量归一化，模型将不得不小心地调整学习率和参数初始化，以适应这种动态分布，这会导致训练变得困难和缓慢。

批量归一化的作用在于解决这个问题，它实时对每个批次的输入数据进行均值和方差的归一化，使每一层的输入分布稳定。这有以下几个优点：

1. **提高训练速度**：BN可以加速训练的收敛，允许使用更高的学习率，从而加快训练速度。
2. **减小过拟合风险**：BN的归一化过程具有正则化效果，有助于降低模型对训练数据的过拟合风险。
3. **简化初始化**：BN减轻了对权重初始化的依赖，使模型更容易初始化，减少了调整的复杂性。
4. **提高模型的泛化能力**：BN允许模型更好地适应不同数据分布，从而提高了模型的泛化能力。

从形式上来说，用$\mathbf{x} \in \mathcal{B}$表示一个来自小批量$\mathcal{B}$的输入，批量规范化$\mathrm{BN}$根据以下表达式转换$\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
$\hat{\boldsymbol{\mu}}_\mathcal{B}$是小批量$\mathcal{B}$的样本均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$是小批量$\mathcal{B}$的样本标准差。
应用标准化后，生成的小批量的平均值为0和单位方差为1。
由于单位方差（与其他一些魔法数）是一个主观的选择，因此我们通常包含*拉伸参数*（scale）$\boldsymbol{\gamma}$和*偏移参数*（shift）$\boldsymbol{\beta}$，它们的形状与$\mathbf{x}$相同。$\boldsymbol{\gamma}$和$\boldsymbol{\beta}$是需要与其他模型参数一起学习的参数。


>LN和BN的对比

![[Pasted image 20231102175001.png]]
批量归一化（Batch Normalization，BN）和层归一化（Layer Normalization，LN）的区别：

1. **应用对象**：
   - 批量归一化（BN）：BN是应用于每个隐藏层的，通常是卷积层或全连接层中的每个特征通道。它的计算基于每个批次中的数据，对每个通道进行独立的归一化。
   - 层归一化（LN）：LN是应用于每个隐藏层的，但不考虑每个批次中的数据。它对每个隐藏层的所有神经元的输出进行独立的归一化。

2. **计算方式**：
   - 批量归一化（BN）：BN通过计算每个特征通道的均值和方差，然后应用归一化和缩放操作。它的计算基于每个批次中的数据。
   - 层归一化（LN）：LN通过计算每个隐藏层的所有神经元的均值和方差，然后应用归一化和缩放操作。它的计算独立于批次。

3. **数据依赖性**：
   - 批量归一化（BN）：BN的归一化过程受每个批次数据分布的影响。因此，如果批次大小较小，BN可能会引入一些噪声。
   - 层归一化（LN）：LN的归一化过程不受批次数据分布的影响，因此在批次大小较小时仍然稳定。

4. **适用场景**：
   - 批量归一化（BN）：BN通常用于卷积神经网络（CNN）和全连接神经网络（DNN）中，对于具有多个特征通道的网络层，尤其有效。
   - 层归一化（LN）：LN通常用于递归神经网络（RNN）或循环神经网络（LSTM、GRU）等序列模型中，其中批次大小可能会受到限制，或者对于每个序列长度较短的情况。





>什么时候用LN，什么时候用BN？

https://www.zhihu.com/question/395811291

深度学习里的正则化方法就是“通过把一部分不重要的复杂信息损失掉，以此来降低拟合难度以及过拟合的风险，从而加速了模型的收敛”。Normalization目的就是让分布稳定下来 (降低各维度数据的方差)

不同正则化方法的区别只是操作的信息维度不同，即选择损失信息的维度不同
在CV中常常使用BN，它是在N维度进行了归一化，而Channel维度的信息原封不动，因为可以认为在CV应用场景中，数据在不同channel中的信息很重要，如果对其进行归一化将会损失不同channel的差异信息。

NLP中不同batch样本的信息关联性不大，而且由于不同的句子长度不同，强行归一化会损失不同样本间的差异信息，所以就没在batch维度进行归一化，而是选择LN，只考虑的句子内部维度的归化。可以认为NLP应用场景中一个样本内部维度间是有关联的，所以在信息归一化时，对样本内部差异信息进行一些损失，反而能降低方差。

总结一下: 选择什么样的归一化方式，取决于你关注数据的哪部分信息。如果某个维度信息的差异性很重要，需要被拟合，那就别在那个维度进行归一化。


## 应用

1.全连接层
将批量规范化层置于全连接层中的仿射变换和激活函数之间。

2.卷积层
在卷积层之后和非线性激活函数之前应用批量规范化。

多个输出通道时，我们需要对这些通道的“每个”输出执行批量规范化，每个通道都有自己的拉伸（scale）和偏移（shift）参数，这两个参数都是标量。
假设我们的小批量包含$m$个样本，并且对于每个通道，卷积的输出具有高度$p$和宽度$q$。
那么对于卷积层，我们在每个输出通道的$m \cdot p \cdot q$个元素上同时执行每个批量规范化。


## 代码——从0实现

```python
import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data


```



```python
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```


应用于LeNet
```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```
## 代码——简明实现

直接用nn中定义好的BatchNorm框架,应用于之前的LeNet。

```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```
通常高级API变体运行速度快得多，因为它的代码已编译为C++或CUDA，而我们的自定义代码由Python实现。


# ResNet
[ResNet（Residual Networks）](https://arxiv.org/abs/1512.03385)是由微软研究员Kaiming He等人于2015年提出的深度卷积神经网络架构。

残差神经网络的主要贡献是发现了“退化现象（Degradation）”，并针对退化现象发明了 “快捷连接（Shortcut connection）”，极大的消除了深度过大的神经网络训练困难问题。

创新点：引入了残差学习，通过添加残差块（Residual Blocks），允许网络在学习过程中专注于学习残差或误差的部分，而不是直接学习完整的映射。这种方法的核心思想是，如果一个恒等映射（将输入直接传递到输出）是我们的目标，那么网络可以很容易地学习残差来逼近这个目标。这使得训练非常深的网络变得更加容易，因为每个残差块只需要学习如何修正前一层的输出，而不是从头开始学习整个映射。

[ResNet-18可视化](https://cwlacewe.github.io/netscope/#/preset/resnet-18-deploy)


## 背景
出现现象:
1. 在2012年的ILSVRC挑战赛中，AlexNet取得了冠军，并且大幅度领先于第二名。由此引发了对AlexNet广泛研究，并让大家树立了一个信念——“越深网络准确率越高”。这个信念随着VGGNet、Inception v1、Inception v2、Inception v3不断验证、不断强化，得到越来越多的认可
2. 通过实验，随着网络层不断的加深，模型的准确率先是不断的提高，达到最大值（准确率饱和），然后随着网络深度的继续增加，模型准确率毫无征兆的出现大幅度的降低。

原因探究：
参考[知乎](https://zhuanlan.zhihu.com/p/101332297)

按道理，层数较多的神经网络，可由较浅的神经网络和恒等变换网络拼接而成。
但是，深度学习的关键特征在于网络层数更深、非线性转换（激活）、自动的特征提取和特征转换，其中，非线性转换是关键目标，它将数据映射到高纬空间以便于更好的完成“数据分类”。随着网络深度的不断增大，所引入的激活函数也越来越多，数据被映射到更加离散的空间，此时已经难以让数据回到原点（恒等变换）。

非线性转换极大的提高了数据分类能力，但是，随着网络的深度不断的加大，我们在非线性转换方面已经走的太远，竟然无法实现线性转换。

于是，ResNet团队在ResNet模块中增加了快捷连接分支，在线性转换和非线性转换之间寻求一个平衡。

## 残差块

以下，左图是正常块，右图是残差块。 
左图虚线框中的部分需要直接拟合出该映射$f(\mathbf{x})$，而右图虚线框中的部分则需要拟合出残差映射$f(\mathbf{x}) - \mathbf{x}$。
![[Pasted image 20231102183911.png]]



ResNet沿用了VGG完整的$3\times 3$卷积层设计：
1. 残差块里首先有2个有相同输出通道数的$3\times 3$卷积层。
2. 每个卷积层后接一个批量规范化层和ReLU激活函数。
3. 跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。

如果想改变通道数，就需要引入一个额外的$1\times 1$卷积层来将输入变换成需要的形状后再做相加运算。

以下是包含或者不包含1X1卷积层的残差块。

![[Pasted image 20231102184915.png]]

下列代码实现Residual Block模块

```python

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 创建一个自定义的残差块（Residual Block）模块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        # 第一个卷积层：3x3 卷积核，填充（padding）为1，步幅（strides）默认为1
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)

        # 第二个卷积层：3x3 卷积核，填充（padding）为1，步幅（strides）默认为1
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # 如果 use_1x1conv 为 True，则添加一个 1x1 卷积层
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # Batch Normalization 层用于规范化输入
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    # 定义前向传播过程
    def forward(self, X):
        # 第一个卷积层，后跟 ReLU 激活函数和批量规范化
        Y = F.relu(self.bn1(self.conv1(X)))

        # 第二个卷积层，后跟批量规范化
        Y = self.bn2(self.conv2(Y))

        # 如果存在第三个卷积层（1x1 卷积），则应用它到输入 X
        if self.conv3:
            X = self.conv3(X)

        # 将第三个卷积层的输出（如果存在）与第二个卷积层的输出相加
        Y += X

        # 最后应用 ReLU 激活函数
        return F.relu(Y)

```


## ResNet

```python
# 定义一个函数 resnet_block，用于创建一个残差块序列
# 输入参数：
#   - input_channels: 输入通道数
#   - num_channels: 卷积层中的输出通道数
#   - num_residuals: 要堆叠的 Residual 模块数量
#   - first_block: 是否为第一个块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []  # 用于存储残差块的列表
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 对于第一个残差块，包含 1x1 卷积和步幅 2 来减小尺寸
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 对于后续的残差块，仅使用 3x3 卷积层
            blk.append(Residual(num_channels, num_channels))
    return blk  # 返回残差块列表


#ResNet构建

# 1.创建一个序列网络 b1，包含初始卷积层、批量规范化、ReLU 和最大池化
# 和GoogleNet一样开始用卷积核汇聚，不一样的是增加了批量规范化。
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 输入通道数 1，输出通道数 64，7x7 卷积核
    nn.BatchNorm2d(64),  # 批量规范化
    nn.ReLU(),  # ReLU 激活函数
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，用于下采样
)

# 创建其他序列网络 b2、b3、b4、b5，每个序列包含一系列残差块
# GoogleNet是采用了4个Inception组成的模块，而ResNet使用的是4个残差块组成的模块。
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 创建主要的神经网络模型 net，将 b1 到 b5 串联在一起，并包括最后的自适应平均池化、展平和全连接层
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化层
                    nn.Flatten(),  # 数据展平
                    nn.Linear(512, 10)  # 全连接层，输出类别数为 10
)


```

ResNet-18：每个模块4个卷积层$\times 4$+第一个$7\times 7$的卷积+全连接层=18层

实战使用：https://pytorch.org/hub/pytorch_vision_resnet/

## Q&A
>Inception块与残差块之间的主要区别是什么？在删除了Inception块中的一些路径之后，它们是如何相互关联的？

**Inception块（Inception Block）**：
1. **结构多样**：Inception块包含多个不同尺寸和类型的卷积核，每个卷积核对输入进行卷积操作。这种多样性有助于网络在不同尺度下捕捉特征，从而提高了网络的性能。
2. **多个路径**：Inception块通常包含不同大小的卷积核，池化操作和1x1卷积，这些操作并行进行，然后将它们的输出连接在一起，形成一个更大的特征图。
3. **参数情况**：Inception块中的卷积操作具有不同的参数，这使得网络的参数量相对较大。

**残差块（Residual Block）**：
1. **残差连接**：残差块通过引入残差连接（shortcut connection）来解决梯度消失问题。残差块的输出是输入与一个残差（差值）的相加，而不是简单的特征映射叠加。
2. **单一路径**：残差块的主要特点是单一路径，通常由两个或更多的卷积层组成，然后将其输入与残差相加。
3. **参数情况**：相对于Inception块，残差块中的卷积操作具有较少的参数，因为它们都是在同一路径上操作的。

如果删除Inception块中的一些路径，这将减少Inception块的多样性和参数数量，可能导致网络的容量减小，因此它们的性能可能会受到影响。在残差块中，如果删除残差连接，将会失去残差块的主要特征，导致网络无法有效地训练。

> 对于更深层次的网络，ResNet引入了“bottleneck”架构来降低模型复杂性。请试着去实现它。

![[Pasted image 20231102191436.png]]

上图左边是基本残差块，右边是Bottleneck结构的残差块。

"Bottleneck" 架构主要特点是在每个残差块中使用了一个"Bottleneck" 结构，它通过引入降低维度的1x1卷积层和增加维度的1x1卷积层来减少网络的复杂性。

"Bottleneck" 架构的残差块关键结构：

1. 1x1 卷积层（降维）：这个卷积层用于减小输入的通道数，以降低计算复杂性。它通常采用1x1卷积核来减小通道数。
2. 3x3 卷积层：这是一个常规的卷积层，用于捕获特征。
3. 1x1 卷积层（增维）：这个卷积层用于增加通道数，以恢复原始输入的通道数，以便将输出与输入进行残差连接。

关于计算和存储的对比运算：

假设输入 feature map 的维度为 256 维，要求输出维度也是 256 维。有以下两种操作：

1. 直接使用 3x3 的卷积核。256 维的输入直接经过一个 3×3×256 的卷积层，输出一个256维的 feature map，那么参数量为：256×3×3×256 = 589824 。
2. 先经过 1x1 的卷积核，再经过 3x3 卷积核，最后经过一个 1x1 卷积核。 256 维的输入先经过一个 1×1×64 的卷积层，再经过一个 3x3x64 的卷积层，最后经过 1x1x256 的卷积层，则总参数量为：256×1×1×64 + 64×3×3×64 + 64×1×1×256 = 69632

可以看到计算量远远小于之前的残差块。

>在ResNet的后续版本中，作者将“卷积层、批量规范化层和激活层”架构更改为“批量规范化层、激活层和卷积层”架构。请尝试做这个改进。


![[Pasted image 20231102192205.png]]

```python
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, stride=1):
        super(Residual, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_channels)  # 批量规范化层
        self.relu = nn.ReLU()  # ReLU 激活层
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=stride, padding=1)  # 卷积层
        self.bn2 = nn.BatchNorm2d(num_channels)  # 批量规范化层
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)  # 卷积层

        # 如果步幅不为1，使用1x1卷积层进行尺寸匹配
        if stride != 1 or input_channels != num_channels:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.bn1(X)
        Y = self.relu(Y)
        Y = self.conv1(Y)
        Y = self.bn2(Y)
        Y = self.relu(Y)
        Y = self.conv2(Y)

        if self.conv3:
            X = self.conv3(X)

        Y += X  # 残差连接
        return Y

```

# DenseNet

https://zhuanlan.zhihu.com/p/141178215
稠密链接网络（DenseNet）是ResNet的逻辑拓展。

## 从ResNet到DenseNet

任意函数在a点的泰勒展开式（Taylor expansion）：
$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \ldots$$

在$a$接近0时，

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$


同样，ResNet将函数展开为

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

也就是说，ResNet将$f$分解为两部分：一个简单的线性项和一个复杂的非线性项。
那么再向前拓展一步，如果我们想将$f$拓展成超过两部分的信息呢？
一种方案便是DenseNet。
![[Pasted image 20231102193541.png]]

DenseNet的主要特点是在网络内部建立了密集的连接模式，使得**每一层的特征图都与前面的所有层直接相连**。这使得每一层都可以获得之前层的信息，包括低级特征和高级特征。这有助于模型学习更丰富和抽象的特征表示。


DenseNet的优点包括：

1. 参数效率：由于特征图的重用，DenseNet具有相对较少的参数，与传统的深度卷积神经网络相比，可以更好地利用有限的数据进行训练。

2. 模型性能：DenseNet在图像分类和物体识别任务中通常能够实现更好的性能，同时还能够减少过拟合的风险。

3. 深度网络训练：密集连接的设计使得梯度更容易传播，有助于训练非常深的神经网络，从而可以从更丰富的特征表示中受益。

## 稠密块（dense Block）

一个*稠密块*由多个卷积块组成，每个卷积块使用相同数量的输出通道。
然而，在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 定义一个基本的卷积块，包括批量归一化、ReLU激活和卷积层
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),  # 批量归一化层
        nn.ReLU(),                      # ReLU激活函数
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)  # 卷积层
    )

# 定义稠密块(Dense Block)模块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)  # 前向传播每个卷积块
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

```


## 过渡层
由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。
而过渡层可以用来控制模型复杂度。
它通过$1\times 1$卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度。
```python
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```


## ResNet


DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层。
```python
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

类似于ResNet使用的4个残差块，DenseNet使用的是4个稠密块
```python
# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]#每个稠密块用四个卷积层
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

最后的网络融合。
```python
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

>为什么我们在过渡层使用平均汇聚层而不是最大汇聚层？


DenseNet的设计目标之一是最大限度地保留输入特征的信息，以便在稠密连接中更好地传播信息。平均汇聚层计算窗口中的平均值，这有助于保留所有特征的信息，而不仅仅是最显著的特征。这对于稠密连接的概念非常重要，因为每个层都可以获得来自前面层的所有信息。

>DenseNet一个诟病的问题是内存或显存消耗过多。为什么？


显存最直接的计算就是一次推断中所产生的所有feature map数目。有些框架会有优化，自动把比较靠前的层的feature map释放掉，所以显存就会减少，或者inplace操作通过重新计算的方法减少一部分显存，但是densenet因为需要重复利用比较靠前的feature map，所以无法释放，导致显存占用过大。

>densenet比resnet参数量少，但训练速度慢的原因分析

https://blog.csdn.net/dulingtingzi/article/details/90514060