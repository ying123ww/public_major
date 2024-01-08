
#   背景
卷积神经网络（CNN）三大特性：区部连接、权值共享、池化。
1. 图像**数据**和**问题**：
   - 图像数据由二维像素网格组成，可以是黑白或彩色（RGB）。
   - 以前的方法将图像数据展平成一维向量，**忽略了图像的空间结构信息**，这在处理图像时不够有效。

2. 卷积神经网络（CNN）：
   - CNN是一种专为**处理图像数据而设计**的神经网络。
   - 现代计算机视觉领域的图像识别、目标检测和语义分割等任务主要基于CNN架构。

3. 设计优势：
   - CNN需要的**参数较少**，容易并行计算，因此在采样和计算效率方面具有优势。

4. 应用领域：
   - CNN不仅在**计算机视觉**中流行，还在处理**一维序列数据（音频、文本、时间序列等）和图结构数据以及推荐系统**中得到广泛应用。

5. CNN的基本元素：
   - 包括**卷积层、填充、步幅、汇聚层、多通道使用**以及现代CNN架构的细节。

6. LeNet模型：
   - 介绍了LeNet模型，这是最早成功应用的CNN之一，起源早于深度学习的兴起。

# 1.why-conv？

处理图像时的归纳偏置：

1. **平移不变性（Translation Invariance）**：

   平移不变性意味着在图像中的对象或特征出现在不同位置时，神经网络的响应应该是相似的。这是因为对于很多视觉任务，我们不关心对象在图像中的确切位置，而是关心其**存在与否以及其特征**。
   例如，在图像分类任务中，无论猫在图像的左上角还是右下角，我们都希望网络能够识别它是一只猫。**平移不变性使网络能够学习到不受对象位置变化的特征。**

2. **局部不变性（Locality）**：

   局部不变性是指神经网络的**前几层应该只关注图像中的局部区域，而不过度强调图像中相隔较远区域的关系**。这是因为图像中的信息通常是局部相关的，而全局关系通常需要更高级的层次来捕捉。通过在前几层网络中引入局部性，网络能够更有效地学习到局部特征，然后在后续层次中聚合这些特征以进行全局决策。

在CNN中，这两个概念通过卷积层和汇聚层的使用得到实现。卷积层通过局部感受野和权重共享实现局部性，而汇聚层通过对局部区域进行池化操作来减小空间尺寸，从而提高网络的计算效率并引入平移不变性。


## 数学公式
### MLP
- 如果输入是一维的：

   对于第一个隐藏层中的第j个神经元（j = 1, 2, ..., $N_h$，其中$N_h$是隐藏层中神经元的数量），计算输出 $z_j$ 如下：

   $$z_j =  b_j+\sum_{i=1}^{N_{in}}(w_{ij} \cdot x_i) $$

   其中：
   - $N_{in}$ 是输入层中神经元的数量。
   - $x_i$ 是输入层中第i个神经元的输出。
   - $w_{ij}$ 是连接输入层的第i个神经元和隐藏层中的第j个神经元的权重。
   - $b_j$ 是隐藏层中的第j个神经元的偏置。
   
   然后，将 $z_j$ 应用激活函数 $f$：

   $$a_j = f(z_j)$$

   $a_j$ 是第j个神经元的激活值，它将作为下一层（或输出层）的输入。

- 如果输入是二维的：
$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

1. $[\mathbf{X}]_{i, j}$：这表示输入图像中位置（$i$, $j$）处的像素值。这是一个二维矩阵，用来表示输入图像的像素值。

2. $[\mathbf{H}]_{i, j}$：这表示隐藏表示（神经元）中位置（$i$, $j$）处的像素值。这也是一个二维矩阵，用来表示神经网络中某一层的隐藏表示。

3. $\mathsf{W}$：这是一个四阶权重张量，用于进行卷积操作。它用于连接输入图像和隐藏表示，以实现信息传递。

4. $\mathbf{U}$：这包含了偏置参数，是与每个隐藏神经元相关的偏置项。

5. $\mathsf{V}$：这是另一个四阶权重张量，通过对$\mathsf{W}$的一种重新索引方式得到。重新索引下标$(k, l)$，使$k = i+a$、$l = j+b$，由此可得$[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$，$\mathsf{V}$用于加权输入图像以计算隐藏表示中的值。

这说明之前图像平面$(k,l)$可以由$(i+a,j+b)$来代替。而这层的图像平面即像素值$[\mathbf{H}]_{i, j}$用$(i,j)$来表示。以上使得$[\mathbf{H}]_{i, j}$是通过在输入图像$\mathbf{X}$中以$(i, j)$为中心，使用权重$[\mathsf{V}]_{i, j, a, b}$对像素进行加权求和得到的。即通过在正偏移和负偏移之间移动来覆盖整个图像来实现的。

再通俗的理解$[\mathsf{V}]_{i, j, a, b}$,就是对于$H$隐藏层每一个像素点$(i, j)$的每一个与上一层对应的偏移$(a, b)$ 的权重值。所以每个像素每个偏移的权重值都不一样，带来了巨大的麻烦。



### 平移不变性的引入
**平移不变性**：在深度学习中，平移不变性是指模型应该能够识别对象的特征，无论这些特征在数据中的位置如何改变。

这个V，U也就是模型需要学到的特征，根据上文的MLP来说，$[\mathsf{V}]_{i, j, a, b}$被$(i,j)$和偏移$(a,b)$所影响。但根据平移不变性，$\mathsf{V}$和$\mathbf{U}$实际上不依赖于$(i, j)$或者$(k,l)$的值，只与它们的偏差(a,b)有关？，即$[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$。并且$\mathbf{U}$是一个常数，比如$u$。因此，我们可以简化$\mathbf{H}$定义为：

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b} [\mathbf{X}]_{i+a, j+b}.$$

这就是*卷积*（convolution）。我们是在使用系数$[\mathbf{V}]_{a, b}$对位置$(i, j)$附近的像素$(i+a, j+b)$进行加权得到$[\mathbf{H}]_{i, j}$。

所以看到，由于平移不变性的引入，实现了权值共享（称权值共享可能还是不是很恰当，因为每一个位置(i,j)对应的a，b的取值范围可能都不一样）？
### 局部性的引入：

局部性。如上所述，为了收集用来训练参数$[\mathbf{H}]_{i, j}$的相关信息，我们不应偏离到距$(i, j)$很远的地方。这意味着在$|a|> \Delta$或$|b| > \Delta$的范围之外，我们可以设置$[\mathbf{V}]_{a, b} = 0$。因此，我们可以将$[\mathbf{H}]_{i, j}$重写为

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


于是就诞生了一个*卷积层*（convolutional layer），$\mathbf{V}$被称为*卷积核*（convolution kernel）或者*滤波器*（filter），亦或简单地称之为该卷积层的*权重*，通常该权重是可学习的参数。

所以又可以看到，权值矩阵由于局部性的引入被规范到了一个正方形，于是诞生了标准的卷积核。

### 多通道卷积的引入：
but图像是一个由高度、宽度和颜色组成的三维张量，比如包含$1024 \times 1024 \times 3$个像素。将$\mathsf{X}$索引为$[\mathsf{X}]_{i, j, k}$。由此卷积相应地调整为$[\mathsf{V}]_{a,b,c}$。
此外，隐藏表示$\mathsf{H}$也最好采用三维张量。对于每一个空间位置，我们想要采用一组而不是一个隐藏表示。得到多个特征。这样一组隐藏表示可以想象成一些互相堆叠的二维网格。
因此，我们可以把隐藏表示H称成为*特征映射*（feature maps），因为每个feature map（即每个通道（channel））都向后续层提供一组空间化的学习特征。直观上可以想象在靠近输入的底层，一些feature map专门识别边缘，而一些feature map专门识别纹理。

为了支持输入$\mathsf{X}$和隐藏表示$\mathsf{H}$中的多个通道，我们可以在$\mathsf{V}$中添加第四个坐标 $d$，即$[\mathsf{V}]_{a, b, c, d}$。综上所述，

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$

这个公式的解释：

1. 输入图像和隐藏表示：
   - $\mathsf{X}$ 表示输入图像，是一个三维张量，包含高度、宽度和颜色通道。
   - $\mathsf{H}$ 表示隐藏表示，也是一个三维张量，与输入图像具有相同的高度和宽度，但包含多个通道。

2. 卷积核和权重矩阵：
   - $\mathsf{V}$ 表示卷积核，它是一个四维张量，包含卷积操作所需的权重参数。第一个维度 $a$ 和第二个维度 $b$ 控制卷积核的空间位置，第三个维度 $c$ 控制输入图像的颜色通道，第四个维度 $d$ 控制输出的隐藏表示通道。

3. 卷积计算过程：
   - 公式中的 $\sum$ 符号表示求和操作。
   - $i$ 和 $j$ 是隐藏表示 $\mathsf{H}$ 中的空间位置索引。
   - $a$ 和 $b$ 分别在 $-\Delta$ 到 $\Delta$ 的范围内进行遍历，用于控制卷积核的平移操作。
   - $c$ 在 $1$ 到 $3$（或更一般地，输入图像的颜色通道数）的范围内进行遍历，以考虑所有输入图像的颜色通道。
   - $d$ 表示卷积核个数。即feature map的个数。即隐藏表示的channel数。
   - $[\mathsf{V}]_{a, b, c, d}$ 表示卷积核 $\mathsf{V}$ 的相应权重参数。
   - $[\mathsf{X}]_{i+a, j+b, c}$ 表示输入图像 $\mathsf{X}$ 中的像素值。

4. 总结：
**多通道卷积过程，应该是输入一张三通道的图片，这时有多个卷积核进行卷积，并且每个卷积核都有三通道，分别对这张输入图片的三通道进行卷积操作。每个卷积核，分别输出三个通道，这三个通道进行求和，得到一个featuremap，有多少个卷积核，就有多少个。**

### 卷积和互相关
什么是卷积？
$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$
可以看到卷积就是，其中一个量翻转平移后，测量两者的重叠。
如果离散化之后，积分即求和：
$$(f * g)(i) = \sum_a f(a) g(i-a).$$
对于二维的张量来说：

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
这和上文的差别在于$(i-a, j-b)$和$(i+a, j+b)$.对于之前推导出来的答案，更应该称之为互相关。至于为什么要证明实际上互相关只是卷积的图像反转，可能是因为更想证明卷积的命名方式也没问题。

本质就是mlp——>（平移不变性+局部性）——>互相关公式。
***
## 一些Q&A
>比较MLP和CNN


多层感知器（Multilayer Perceptron，MLP）和卷积神经网络（Convolutional Neural Network，CNN）是深度学习中两种常用的神经网络架构。以下是它们的主要区别和特点：

1. 数据类型：
   - MLP：通常用于处理结构化数据，如表格数据、文本数据和时间序列数据。MLP的每个神经元都与前一层的每个神经元连接，适合处理**全连接的数据**。
   - CNN：主要用于处理**图像和视频数据**，也可用于处理**一维数据，如音频信号**。CNN在处理图像等二维数据时具有出色的性能，因为它可以捕捉空间结构。

2. 网络结构：
   - MLP：MLP是全连接的神经网络，每一层的神经元都与前一层的每个神经元连接。通常由**输入层、隐藏层和输出层**组成。
   - CNN：CNN使用**卷积层、池化层和全连接层的组合**。**卷积层用于提取特征，池化层用于减小空间维度，全连接层用于输出分类结果。**

3. 参数共享：
   - MLP：在MLP中，每个神经元与前一层的每个神经元连接，没有参数共享。
   - CNN：CNN的卷积层使用参数共享的方式，每个卷积核在整个输入上滑动，从而可以**检测相似的特征在不同位置的存在**。

4. 适用领域：
   - MLP：适用于分类、回归和一般的监督学习任务。结构化数据。
   - CNN：在**图像分类、目标检测、图像分割和处理具有空间结构的数据方面表现出色**。CNN也可用于文本分类等一维数据。非结构化数据。


> 假设卷积层(6.1.3)覆盖的局部区域delta=0。在这种情况下，证明卷积内核为每组通道独立地实现一个全连接层。

 实际就是问，1×1的卷积核是否等价于全连接?
 [答](https://www.zhihu.com/question/274256206)：我个人还不是很理解？
 具体可以看[https://www.jiqizhixin.com/articles/2019-02-22-22](https://www.jiqizhixin.com/articles/2019-02-22-22)

>为什么平移不变性可能也不是好主意？

答：论文 [https://arxiv.org/pdf/1805.12177.pdf](https://arxiv.org/pdf/1805.12177.pdf) 提出了一个观点，即当处理小尺寸的图像并且发生平移时，CNN可能会出现识别错误的现象。这表明卷积层的平移不变性并不总是有效的，尤其对于小尺寸图像。

> 当从图像边界像素获取隐藏表示时，我们需要思考哪些问题？

答：需要考虑是否填充padding，以及填充多大的padding的问题

>描述一个类似的音频卷积层的架构。

一种基于卷积神经网络的音频特征生成方法，首先对声音信号进行预处理和离散傅里叶变换计算声音信号的幅度谱，形成二维谱图信号；然后搭建以上述二维谱图信号为输入的一维卷积神经网络并进行模型训练，得到特征生成器模型；最后对待测声音进行预处理和离散傅里叶变换得到二维谱图信号，并将其送入训练好的一维卷积神经网络，通过卷积网络计算，得到输出即为所要生成的音频特征，实现声音信号的音频特征生成。

>卷积层也适合于文本数据吗？为什么？

[https://zhuanlan.zhihu.com/p/34558743](https://zhuanlan.zhihu.com/p/34558743)
也可以结合后续的nlp。



 # 2. 图像卷积

## 互相关的计算
点对点的计算。
输出大小等于输入大小[$n_h \times n_w$]减去卷积核大小[$k_h \times k_w$]，即：

$$(n_h-k_h+1) \times (n_w-k_w+1).$$
 

```python
import torch
def corr2d(X, K):
    # 函数用于计算二维互相关运算

    # 获取卷积核的高度和宽度
    h, w = K.shape
    # 创建一个全零矩阵，用于存储互相关运算的结果
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 遍历输出矩阵的每个元素
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 计算互相关运算的结果并存储在输出矩阵中
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    # 返回互相关运算的结果
    return Y

```

```python
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```
## 构造卷积层
卷积层中两个主要参数：卷积核and标量偏置。所以在在`__init__`构造函数中，将`weight`和`bias`声明为两个模型参数。

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

高度和宽度分别为$h$和$w$的卷积核可以被称为$h \times w$卷积
## 应用——边缘检测
边缘检测：通过找到像素变化的位置来检测图像中不同颜色的边缘。

```python
#1.构造像素
X = torch.ones((6, 8))
X[:, 2:6] = 0

  
#2.构造卷积核
K = torch.tensor([[1.0, -1.0]])

#互相关运算
Y = corr2d(X, K)
```

结果为：
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])

但是将输入的二维图像转置，再进行如上的互相关运算，之前检测到的垂直边缘就会消失。
说明这个**卷积核`K`只可以检测垂直边缘**，无法检测水平边缘。

## 卷积核的学习
上文是用已知卷积核去检测垂直黑白边缘，但是如果对象更加复杂，我们就要让卷积核自己去学习。比如学习由`X`生成`Y`中的卷积核。

方法：
1. 构造一个卷积层，并将其卷积核初始化为随机张量。
2. 我们比较`Y`与卷积层输出的平方误差，然后计算**梯度来更新卷积核**。

为了简单起见，使用内置的二维卷积层，并忽略偏置。

```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))
```

tensor([ [ 1.0003, -0.9699]])
可以看到针对这张图，结果和我们之前采用的卷积核K非常的相似。

## 一些Q&A

1. 在我们创建的`Conv2D`自动求导时，有什么错误消息？
意思就是把之前的nn.Conv2d换成自己设计的Conv2D。
```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
#conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
conv2d = Conv2D( kernel_size=(1, 2))
```
错误信息：
{
	"name": "RuntimeError",
	"message": "The size of tensor a (0) must match the size of tensor b (7) at non-singleton dimension 3",
	"stack": "---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
d:\\master\\study_task\\first_level(8.19-11)\\d2l-zh\\pytorch\\chapter_convolutional-neural-networks\\conv-layer.ipynb 单元格 17 line 1
     <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb#X22sZmlsZQ%3D%3D?line=9'>10</a> for i in range(10):
     <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb#X22sZmlsZQ%3D%3D?line=10'>11</a>     Y_hat = conv2d(X)
---> <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb#X22sZmlsZQ%3D%3D?line=11'>12</a>     l = (Y_hat - Y) ** 2
     <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb#X22sZmlsZQ%3D%3D?line=12'>13</a>     conv2d.zero_grad()
     <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb#X22sZmlsZQ%3D%3D?line=13'>14</a>     l.sum().backward()

RuntimeError: The size of tensor a (0) must match the size of tensor b (7) at non-singleton dimension 3"
}

只要把X，Y的形式改了就行
```python
X = X.reshape((6, 8))
Y = Y.reshape(( 6, 7))
```

# 3.填充padding &步幅 stride

 [CNN中stride（步幅）和padding（填充）的详细理解_cnn stride-CSDN博客](https://blog.csdn.net/weixin_42899627/article/details/108228008)
**填充：在输入特征图的每一边添加一定数目的行列，使得输出的特征图的长、宽 = 输入的特征图的长、宽**
**步幅：卷积核经过输入特征图的采样间隔**

设置填充的目的：希望每个输入方块都能作为卷积窗口的中心
设置步幅的目的：希望减小输入参数的数目，减少计算量

通常，当垂直步幅为$s_h$、水平步幅为$s_w$时，输出形状为
$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$
即
$$\lfloor(n_h-k_h+p_h)/s_h+1\rfloor \times \lfloor(n_w-k_w+p_w)/s_w+1\rfloor.$$
使输入输出具有相同的尺寸：设置卷积核$p_h=k_h-1$和$p_w=k_w-1$，则输出形状将简化为$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。

更进一步，如果输入的高度和宽度可以被垂直和水平步幅整除，则输出形状将为$(n_h/s_h) \times (n_w/s_w)$。

为了简洁起见，当输入高度和宽度两侧的**填充数量**分别为$p_h$和$p_w$时，我们称之为填充$(p_h, p_w)$。当$p_h = p_w = p$时，填充是$p$。
当高度和宽度上的步幅分别为$s_h$和$s_w$时，我们称之为步幅$(s_h, s_w)$。特别地，当$s_h = s_w = s$时，我们称步幅为$s$。默认情况下，填充为0，步幅为1。在实践中，我们很少使用不一致的步幅或填充，也就是说，我们通常有$p_h = p_w$和$s_h = s_w$。

# 4.多输入多输出通道

## 多输入
多通道**输入**和多输入通道**卷积核**之间进行**二维互相关运算**：由于输入和卷积核都有$c_i$个通道（三维），我们可以对每个通道输入的二维张量和卷积核的二维张量进行互相关运算，再对**通道求和（将$c_i$的结果相加）得到二维张量**。
![[Pasted image 20231101174733.png]]

简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。

## 多输出
因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器。 为了检测不同的特征，应该运用**多个卷积核**。


用$c_i$和$c_o$分别表示输入和输出通道的数目，并让$k_h$和$k_w$为卷积核的高度和宽度。

输入对应卷积核$c_i\times k_h\times k_w$，为了有$c_o$的输出，则要有$c_o$个卷积核。

# 1X1卷积核

$1\times 1$卷积的唯一计算发生在通道上。
输出中的每个元素都是从输入图像中同一位置（不同通道）的元素的线性组合。从通道数ci变成co。

作用：
- 可以升降维度，创造更多（少）的feature map
- $1\times1$卷积虽然没法捕捉空间上的pattern，但是可以捕捉深度上的pattern
- 一般和其他卷积层组合使用，比如说一对([1 x 1, 3 x 3],[1 x 1, 5 x 5])卷积层能想象成一个卷积层，能捕捉更复杂的pattern

  
## 一些Q&A


1. 假设我们有两个卷积核，大小分别为$k_1$和$k_2$（中间没有非线性激活函数）。

    1. 证明运算可以用单次卷积来表示。
	答：根据卷积结合律，当然可以。
    1. 这个等效的单个卷积核的维数是多少呢？
	答:$k_1$+$k_2$-1
    1. 反之亦然吗？

1. 假设输入为$c_i\times h\times w$，卷积核大小为$c_o\times c_i\times k_h\times k_w$，填充为$(p_h, p_w)$，步幅为$(s_h, s_w)$。

1. 前向传播的计算成本（乘法和加法）：
   
   - 输出的高度为 $h_o = \frac{h + 2p_h - k_h}{s_h} + 1$。
   - 输出的宽度为 $w_o = \frac{w + 2p_w - k_w}{s_w} + 1$。

对于每个输出通道（$c_o$），计算成本为：

   - 乘法次数：$c_o \times c_i \times k_h \times k_w \times h_o \times w_o$
   - 加法次数：$c_o \times h_o \times w_o$

2. 内存占用：
   
   内存占用取决于输入数据、卷积核、输出数据的大小。对于输入数据、卷积核和输出数据，它们需要存储的元素数量分别是：

   - 输入数据：$c_i \times h \times w$
   - 卷积核：$c_o \times c_i \times k_h \times k_w$
   - 输出数据：$c_o \times h_o \times w_o$

   这些元素的大小也取决于数据类型（例如，32位浮点数或16位整数）。

# 5.汇聚层

*汇聚*（pooling）层目的：1.降低卷积层对位置的敏感性。2.降低对空间降采样表示的敏感性。

汇聚层不包含参数，汇聚窗口从输入张量的左上角开始，从左往右、从上往下的在输入张量内滑动。在汇聚窗口到达的每个位置，它计算该窗口中输入子张量的最大值（最大汇聚）或平均值（平均汇聚）。

在处理多通道输入数据时，[**汇聚层在每个输入通道上单独运算**]，而不是像卷积层一样在通道上对输入进行汇总。
这意味着汇聚层的输出通道数与输入通道数相同。

# 6.Lenet

1989年提出，当时目的是识别图像中的手写文字。

## 结构
总体来看，**LeNet（LeNet-5）由两个部分组成：**
* 卷积编码器：由两个卷积层组成;
* 全连接层密集块：由三个全连接层组成。
![[Pasted image 20231101180728.png]]
每个卷积块中的基本单元：一个卷积层、一个sigmoid激活函数和平均汇聚层。

每个卷积层使用$5\times 5$卷积核和一个sigmoid激活函数。这些层将输入映射到多个二维特征输出，通常同时增加通道的数量。第一卷积层有6个输出通道，而第二个卷积层有16个输出通道。每个$2\times2$池操作（步幅2）通过空间下采样将维数减少4倍。

  ```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

通过代码验证模型结构：
Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
Flatten output shape: 	 torch.Size([1, 400])
Linear output shape: 	 torch.Size([1, 120])
Sigmoid output shape: 	 torch.Size([1, 120])
Linear output shape: 	 torch.Size([1, 84])
Sigmoid output shape: 	 torch.Size([1, 84])
Linear output shape: 	 torch.Size([1, 10])

## 实例验证——Fashion-MNIST

```python
import torch
from torch import nn
from d2l import torch as d2l

# 创建一个神经网络模型
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 第一个卷积层，使用Sigmoid激活函数
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第一个平均池化层
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 第二个卷积层，使用Sigmoid激活函数
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第二个平均池化层
    nn.Flatten(),  # 展平操作，将多维数据展平为一维
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),  # 全连接层1，使用Sigmoid激活函数
    nn.Linear(120, 84), nn.Sigmoid(),  # 全连接层2，使用Sigmoid激活函数
    nn.Linear(84, 10))  # 输出层

# 设置批量大小
batch_size = 256

# 加载Fashion MNIST数据集的训练集和测试集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None): 
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) #X初始化
    net.apply(init_weights)  # 初始化模型的权重
    print('training on', device)
    net.to(device)  # 将模型移到指定的设备（GPU）
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 使用随机梯度下降优化器
    loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    #之前保存的画图函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()  # 设置模型为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 清零梯度
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 前向传播计算预测
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            #l * X.shape[0]：这部分用于将当前批次的损失乘以批次大小，以便获得批次中所有样本的总损失。
            #d2l.accuracy(y_hat, y)：这部分用于计算当前批次的模型预测准确率。
            #X.shape[0]：这是批次的大小，表示当前批次中有多少个样本。
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

            #以下都是画图和计算准确率等
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 在测试集上计算精度
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

# 设置学习率和训练周期数
lr, num_epochs = 0.9, 10

# 使用GPU训练模型
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

```

loss 0.462, train acc 0.827, test acc 0.782
38743.2 examples/sec on cuda:0
![[Pasted image 20231101183207.png]]

总体步骤：
这段代码是用于训练一个深度学习模型的步骤，以下是代码的总体步骤：

1. 创建神经网络模型 `net`，这个模型包括卷积层、池化层和全连接层。
2. 设置批量大小 `batch_size` 并加载Fashion MNIST数据集的训练集和测试集。
3. 定义了一个用于在GPU上计算模型在数据集上的精度的函数 `evaluate_accuracy_gpu`，该函数在每个训练周期结束后用于评估模型的性能。
4. 定义了一个用于在GPU上训练模型的函数 `train_ch6`，该函数包括了以下步骤：

   a. 初始化模型的权重，使用了 Xavier 初始化方法，以确保模型参数的初始值合理。
   b. 将模型移到指定的设备（GPU）。
   c. 设置优化器（使用随机梯度下降 SGD）和损失函数（交叉熵损失）。
   d. 创建一个用于可视化训练进程的 `animator`。
   e. 在每个训练周期中，遍历训练数据集，进行以下操作：

      - 清零梯度，以准备接收新的梯度更新。
      - 前向传播，计算模型的预测。
      - 计算损失并进行反向传播，以计算梯度和更新模型参数。
      - 更新统计指标，包括训练损失和训练准确率。
      - 更新可视化图表以显示训练损失和训练准确率。

   f. 在每个训练周期结束后，使用 `evaluate_accuracy_gpu` 函数在测试集上计算模型的测试准确率，并更新可视化图表。

5. 最后，打印出最终的训练损失、训练准确率和测试准确率，以及训练速度（每秒处理的样本数）。

## 7.CNN可视化

https://poloclub.github.io/cnn-explainer/