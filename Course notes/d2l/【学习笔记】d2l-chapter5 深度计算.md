
# 目的
深入探索深度学习计算的关键组件：

 - 模型构建
 - 参数访问和初始化
 - 设计定义层和块
 - 模型读写到磁盘
 - 利用GPU实现显著性加速
 
 # 基础知识
 ## 层和块

1. 层（Layers）：
   - 层接收一组输入，生成相应输出，由一组可调参数实现。
   - 层是神经网络模型的基本组成单元，它执行特定的数学操作或变换。这些操作通常包括线性变换、激活函数、归一化等。
   - 常见的神经网络层包括全连接层（全连接神经网络层）、卷积层（用于卷积神经网络CNNs）、循环层（用于循环神经网络RNNs）等。

3. 块（Blocks）：
   - 块是将多个层组合在一起的方式，以构建更大的、有层次结构的模型部分。
   - 块可以有额外的逻辑，例如跳跃连接、循环结构、条件分支等，用于实现更复杂的神经网络拓扑结构。

总结：层是神经网络的基本构建块；块是将多个层组合在一起，以构建更大、更复杂的神经网络模型部分。

# 1.编程角度-构建块


> 自定义块和顺序块的区别？

在PyTorch或类似的深度学习框架中，"自定义块"和"顺序块"是两种不同的模型构建方法。

1. 自定义块（Custom Block）：
   - 自定义块是指自己编写的、具体用途的模块或层。
   - 自由定义和组合不同的层、操作和逻辑来创建自定义块，以满足特定需求。
   - 自定义块的定义通常涉及到编写一个新的类，该类继承自`nn.Module`，并重写`__init__`和`forward`方法。
   - 自定义块提供了极大的灵活性，允许按照自己的要求构建模型。

示例：在前一回答中的MLP模型中，`self.hidden`和`self.out`都是自定义块。

2. 顺序块（Sequential Block）：
   - 顺序块是一种组合已存在层或自定义块的方法，按照顺序排列它们以构建一个完整的模型。
   - 顺序块是由框架提供的高级构建块，通常用于顺序连接多个层或块，以构建更复杂的模型。
   - 这种方法非常适合简单的线性堆叠模型，其中层按顺序堆叠。
   - 通过使用顺序块，可以更紧凑地定义和构建模型。

示例：在PyTorch中，`nn.Sequential`是一个常用的顺序块，它可以按顺序包含各种层或自定义块。


### 自定义块

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```
符合上文：自定义块的定义通常涉及到编写一个新的类，该类继承自`nn.Module`，并重写`__init__`和`forward`方法。

使用：

```python
net = MLP()
net(X)
```


***
**执行控制流：**
这种控制流的使用允许在神经网络中执行非线性操作，例如根据输入数据的不同条件采取不同的计算步骤，或者迭代操作以**改变输入数据**。
且在某些情况下，模型需要处理常数参数（constant parameters），这些参数在优化过程中不会更新。例如，文本中提到的 FixedHiddenMLP 类用于处理常数参数的层，其中常数参数是指定的常量 c，而不是需要通过反向传播进行更新的参数。

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，初始化后为常量。因此其在训练期间保持不变，不被反向传播更新
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```
常量参数：
requires_grad=False 表示创建的张量不需要计算梯度，也就是说它是一个常数，不会参与反向传播的梯度计算。

控制流：
在上述代码中，作者提到的 "在前向传播函数中执行 Python 的控制流" 意味着在神经网络的前向传播过程中，可以执行一些常规的 Python 控制流语句，如 `if`、`while` 等，以根据输入数据或其他条件对网络的操作进行自定义控制。

在示例代码中，`FixedHiddenMLP` 类的 `forward` 方法包含了多个控制流元素：

1. 使用了 `while` 循环，其条件是 `X.abs().sum() > 1`，（L1范数>1）只要满足这个条件，就会一直执行循环体内的操作。这是一种对输入数据 `X` 进行迭代处理的控制流。

2. 在循环体内，执行了 `X /= 2` 操作，对输入 `X` 进行除以 2 的操作。这是一种逐步缩小输入的控制流操作。

3. 最终，返回了 `X.sum()`，对输入数据的求和结果。


### 顺序块
典型会用到的顺序块：
```python
import torch.nn as nn

# 定义一个简单的神经网络模型
model = nn.Sequential(
    nn.Linear(784, 128),  # 输入层（784维）到隐藏层（128维）的全连接层
    nn.ReLU(),            # 隐藏层的激活函数
    nn.Linear(128, 10)    # 隐藏层（128维）到输出层（10维）的全连接层
)

# 打印整个模型的结构
print(model)
# 构建
model(X)

```


> 什么是nn.Sequential？

 `nn.Sequential` 和它的工作原理：

1. `nn.Sequential` 是 PyTorch 中表示一个块（block）的类，它继承自 `nn.Module`，因此它本身也是一个模块。它的主要功能是维护一个由模块（`Module`）组成的有序列表。

2. 在 `nn.Sequential` 中，可以将不同的模块（例如全连接层、激活函数等）按顺序组成一个模型块，这个顺序会影响前向传播的流程。

3. 前向传播函数是 `nn.Sequential` 的核心。当调用模型对象（例如 `net(X)`）时，实际上是在调用前向传播函数 `net.__call__(X)`。这个前向传播函数非常简单，它将列表中的每个模块按照顺序连接在一起，将每个模块的输出作为下一个模块的输入。

> nn.Linear维度如何计算？

如果输入数据的维度表示为 (32, 3, 224, 224)，这意味着有一个批处理大小为 32 的数据，每个数据样本是一个彩色图像，具有 3 个通道（红、绿、蓝），每个通道的图像分辨率为 224x224 像素。

- 32 表示批处理的大小，即一次处理的图像数量。
- 3 表示图像的通道数，通常是红、绿和蓝通道。
- 224 表示图像的高度。
- 224 表示图像的宽度。

这种维度表示非常常见，特别是在卷积神经网络（CNNs）中，因为它适用于图像数据。

如果需要，则需要将数据扁平化放入：

```python
import torch.nn as nn

# 定义全连接层
fc_layer = nn.Linear(3 * 224 * 224, 128)

# 假设输入数据是 input_data，维度为 (32, 3, 224, 224)
input_data = input_data.view(32, -1)  # 使用view将数据扁平化

# 将扁平化后的数据传递给全连接层
output = fc_layer(input_data)

```

***
将与默认`Sequential`类功能相似的`MySequential`类写出：

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```
> 存储块的方式:_module的类型是OrderDict，什么意思？

`_modules` 是一个 `OrderedDict`（有序字典）类型的成员变量，它是 PyTorch 中用于存储模型的子模块的容器。`OrderedDict` 是 Python 标准库 `collections` 模块中的一种数据结构，它与普通字典（`dict`）类似，但与普通字典不同，它记住了字典中元素的添加顺序。

具体来说，`OrderedDict` 有以下特点：

1. **顺序保持**：与普通字典不同，`OrderedDict` 会保持元素添加的顺序。这意味着按照元素添加的顺序遍历它们。

2. **可迭代**：可以像迭代普通字典一样遍历 `OrderedDict` 中的元素。

3. **有序性**：`OrderedDict` 的有序性对于模型定义和前向传播很重要，因为它确保了模块的顺序以及在前向传播中按照正确的顺序执行模块。在深度学习中，`OrderedDict` 常用于存储模型的子模块，如卷积层、全连接层等。这有助于确保模块按照添加的顺序执行。


总之，`OrderedDict` 是一种有序的字典数据结构，用于存储模型的子模块，确保模块的顺序和前向传播的正确性。

> 如果将`MySequential`中存储块的方式(_module)更改为Python列表，会出现什么样的问题？

可能会引发一些问题：

1. **遍历顺序问题**：Python 列表是一种无序的数据结构，它不保证块的顺序。这意味着在前向传播时，无法保证块的顺序，因此模型的行为可能会出现问题。

2. **模块的唯一标识问题**：在 `_modules` 中，每个模块都有一个唯一的名称[fc1,fc2]，这有助于标识每个模块。如果使用 Python 列表，可能需要自行处理模块的标识，以确保它们在前向传播中以正确的顺序执行。

3. **模型保存和加载问题**：PyTorch 的模型保存和加载机制依赖于 `_modules` 中的模块有唯一的名称。如果使用 Python 列表，可能会导致模型的保存和加载出现问题。

##  组合块

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```


1. `chimera` 是一个顺序组合块，它包含了三个部分：首先是一个 `NestMLP` 的实例，然后是一个线性层，最后是一个 `FixedHiddenMLP` 的实例。

2. 通过 `chimera(X)` 对输入数据 `X` 进行前向传播。在前向传播过程中，输入数据 `X` 依次通过 `NestMLP`、线性层、`FixedHiddenMLP`，每个部分的输出作为下一个部分的输入。

## 平行快
>实现一个块，它以两个块为参数，例如`net1`和`net2`，并返回前向传播中两个网络的串联输出。这也被称为平行块。

- Q1：平行快和顺序块的区别
平行块（Parallel Block）和顺序块（Sequential Block）是两种不同的控制流结构，它们在程序执行中有明显的区别：

1. 顺序块（Sequential Block）：
   - 顺序块中的操作或任务按照它们的顺序依次执行，一个操作完成后才会执行下一个操作。
   - 顺序块通常用于串行执行任务，其中一个操作的输出通常是下一个操作的输入。
   - 例如，如果你有一系列处理数据的操作，它们必须按照指定的顺序执行，那么你可以使用顺序块来组织这些操作。

2. 平行块（Parallel Block）：
   - 平行块中的操作或任务可以同时执行，而不必等待前一个操作完成。
   - 平行块通常用于处理并行任务，其中多个操作可以同时执行，从而提高程序的性能和并行性。
   - 例如，如果你有多个独立的操作，它们不依赖于彼此的结果，那么你可以使用平行块来同时执行它们，以加快整体处理速度。

总结：顺序块强调操作的串行执行，一个接一个地按顺序执行，而平行块强调操作的并行执行，多个操作可以同时进行而不相互阻塞。选择使用哪种块取决于任务的性质和需求，以及你希望控制程序执行的方式。

 - Q2：根据问题，那为什么平行块也可以返回两个网络的串联输出？

平行块是指多个操作可以并行执行，但你仍然可以选择如何处理它们的输出，包括串联或其他方式的组合，具体取决于你的设计和任务要求。
***
解决问题代码：

使用深度学习框架（如PyTorch）中的相应功能来串联两个网络的输出。实例函数：接受两个神经网络（net1和net2）作为参数，并返回它们前向传播的串联输出：

```python
import torch
import torch.nn as nn

class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super(ParallelBlock, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        output1 = self.net1(x)
        output2 = self.net2(x)
        concatenated_output = torch.cat((output1, output2), dim=1)  # 按维度1串联输出
        return concatenated_output
```

在上面的代码中，我们定义了一个名为`ParallelBlock`的自定义模块，它将两个网络`net1`和`net2`作为参数传递给构造函数。在`forward`方法中，我们首先分别对输入`x`使用这两个网络，然后使用`torch.cat`函数按维度1（通常是通道维度）串联它们的输出。
可以对比顺序快，在forward中是顺序连接，for循环。

使用`ParallelBlock`：

```python
# 创建两个示例网络
net1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

net2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

# 创建ParallelBlock实例
parallel_block = ParallelBlock(net1, net2)

# 输入数据
input_data = torch.randn(1, 3, 32, 32)  # 示例输入数据

# 调用ParallelBlock的forward方法获取串联输出
output = parallel_block(input_data)
```

上述示例将`net1`和`net2`的输出串联在一起，并返回`output`，其中`output`的通道数是两个网络输出的通道数之和。

## 多块
> 假设我们想要连接同一网络的多个实例。实现一个函数，该函数生成同一个块的多个实例，并在此基础上构建更大的网络。

使用PyTorch创建多个相同模型块的实例，并将它们连接在一起以构建更大的网络：

```python
import torch
import torch.nn as nn

# 定义一个简单的模型块
class SimpleBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义一个函数，生成多个模型块的实例，并将它们连接在一起
def build_large_network(num_blocks, input_dim, hidden_dim, output_dim):
    blocks = nn.ModuleList([SimpleBlock(input_dim, hidden_dim, hidden_dim) for _ in range(num_blocks)])
    output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(x):
        for block in blocks:
            x = block(x)
        x = output_layer(x)
        return x
    
    return forward

# 使用示例：
input_dim = 64
hidden_dim = 128
output_dim = 10
num_blocks = 3  # 生成3个相同的模型块

large_network = build_large_network(num_blocks, input_dim, hidden_dim, output_dim)
print(large_network)

# 之后可以将large_network用于训练和预测
```

首先定义了一个简单的模型块（SimpleBlock），然后编写了一个函数（build_large_network），该函数创建多个模型块的实例，并将它们连接在一起。

> 为什么用了nn.ModuleList，而不是前文讨论的_modules()？

首先，nn.ModuleList 是PyTorch中的一个有用工具，用于管理模型中的子模块。它与Python的标准列表略有不同。所以前文说的不用python列表依旧成立。

其次，`_modules()` 方法是`nn.Module`的一个内部方法，用于获取包含在模块中的所有子模块。可以使用这个方法来访问子模块，但它没有提供`nn.ModuleList`那样的自动参数注册、方便的迭代和模型保存/加载功能。

主要的区别是：

1. 参数注册：`nn.ModuleList` 内部的子模块会被自动注册，这意味着它们的参数会自动包含在模型的参数列表中，而且很容易地使用`model.parameters()`来获取所有参数。使用 `_modules()` 方法时，需要手动注册子模块的参数，这可能导致出错或不便。

2. 方便的迭代：`nn.ModuleList` 可以像列表一样轻松迭代访问子模块，而 `_modules()` 方法返回一个字典，需要编写额外的代码来进行迭代操作。

3. 模型的保存和加载：`nn.ModuleList` 内部的子模块可以正确地保存和加载，而 `_modules()` 方法返回的字典中没有这种内置的支持，因此需要手动处理模型的保存和加载。

总之，`nn.ModuleList` 提供了更便捷和直观的方式来组织和管理模型中的子模块，而 `_modules()` 方法更适用于更底层的操作和特定的用例。在大多数情况下，使用 `nn.ModuleList` 是更好的选择，因为它提供了更高级的功能和易用性。

 - Q1：什么是参数注册？[具体答案](https://blog.csdn.net/luo3300612/article/details/97815207)
 
 所谓的注册，就是当参数注册到这个网络上时，它会随着在外部调用net.cuda()后自动迁移到GPU上，而没有注册的参数则不会随着网络迁到GPU上，这就可能导致输入在GPU上而参数不在GPU上.文中很好的体现了nn.ModuleList是能够自动注册的。
 - Q2：之前的MySequential可以用nn.ModuleList重写 `_modules` 字典吗？
 
 之前的代码:
 

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

答：

当然，可以使用`nn.ModuleList` 来重写 `_modules` 字典。不过，代码进行一些调整。`nn.ModuleList` 是一个容器，用于存储 `nn.Module` 对象的列表，但它不提供与 `_modules` 相同的字典式访问方式。以下是使用 `nn.ModuleList` 重写你的 `MySequential` 类的方式：

```python
import torch.nn as nn

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module_list = nn.ModuleList(args)

    def forward(self, X):
        for block in self.module_list:
            X = block(X)
        return X
```

在这个版本中，我们使用 `nn.ModuleList` 来存储作为参数传递给构造函数的模块。

 - Q3：这两种方法有无优劣之分？`nn.ModuleList` & `_modules` 字典
 这两种方法都用于构建一个自定义的Sequential模块，允许你按顺序堆叠多个子模块。它们的功能基本相同，但有一些微小的区别：

1. 原始方法（使用 OrderedDict）：

   这个方法使用了一个 OrderedDict 来保存子模块，并依赖于添加子模块的顺序来决定它们的执行顺序。这意味着你可以通过添加子模块的顺序来定义前向传播的执行顺序。这种方法比较直观，因为你可以清晰地看到每个子模块是如何按照添加的顺序依次执行的。

2. 使用 nn.ModuleList 方法：

   这个方法使用 nn.ModuleList 来存储子模块，它允许你像列表一样管理子模块，但不依赖于添加顺序来决定执行顺序。相比于 OrderedDict，这种方法在某些情况下可能更具灵活性，可以轻松地重排子模块的顺序，而无需更改添加顺序。

# 2.参数管理和初始化
* 访问参数，用于调试、诊断和可视化；
* 参数初始化；
* 在不同模型组件间共享参数。

## 访问参数
已经创建好块之后：
```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```
访问参数：

```python
print(net[2].state_dict())
```
输出：OrderedDict([('weight', tensor([ [-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, -0.0334, -0.2263]])), ('bias', tensor([0.0887]))])

这个全连接层包含两个参数，分别是该层的权重和偏置。两者都存储为单精度浮点数（float32）。
注意，参数名称允许唯一标识每个参数，即使在包含数百个层的网络中也是如此。

 1. 访问目标参数

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([0.0887], requires_grad=True)
tensor([0.0887])



 

2. 一次性访问所有参数

访问第一个全连接层的参数和访问所有层
```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))

从而提供了另一种访问网络参数的方式
```python
net.state_dict()['2.bias'].data
```

3. 从嵌套块收集参数

首先生成嵌套块：
```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```
其次查看网络结构 ：print（rgnet）


```bash
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)

```

于是调用的时候可以采用类似嵌套列表索引的方式来访问层

```python
rgnet[0][1][0].bias.data
```

第0，第1，第0:[Linear(in_features=4, out_features=8, bias=True)]的偏置数据。


## 参数初始化
### 背景

>为什么参数初始化很重要？


1. 避免梯度消失或梯度爆炸：**神经网络的训练依赖于梯度下降算法，如果参数初始化不合适，梯度可能会变得过小（梯度消失）或过大（梯度爆炸）**，导致网络无法有效地学习。合适的初始化可以有助于缓解这些问题，帮助网络更快地收敛。

2. 提高收敛速度：合适的参数初始化可以使网络更快地达到收敛，减少训练时间。如果参数初始化不当，网络可能需要更多的迭代才能学习到有效的特征表示。

3. 改善泛化能力：好的参数初始化方法有助于提高模型的泛化能力，使其在未见过的数据上表现更好。这是因为合适的初始化可以帮助网络更好地学习数据的统计特性，而不仅仅是记住训练数据。

4.  防止权重对称性：如果**所有的权重初始值都相同，那么在网络的前向传播和反向传播过程中，神经元的行为将高度对称，这会限制网络的表达能力**。合适的初始化可以破坏权重对称性，使网络更能够学习复杂的特征。

5. 调整学习率：某些初始化方法可以根据网络的深度和结构来自动调整学习率，从而更好地适应不同层的训练需求。

一些常见的参数初始化方法包括随机初始化（如Xavier初始化、He初始化）、预训练模型的参数初始化、以及使用特定领域知识的自定义初始化方法。

> 有哪些初始化方法？

在深度学习中，有许多不同的参数初始化方法，其中一些常见的包括：

1. **随机初始化（Random Initialization）**：这是最常见的初始化方法之一，它为每个权重参数随机分配一个小的值，通常是从均匀分布或正态分布中采样。随机初始化可以帮助打破权重对称性，启动神经网络的学习过程。常见的随机初始化包括使用均匀分布（在[-a, a]范围内采样）或正态分布（均值为0，标准差为a）。

2. **零初始化（Zero Initialization）**：将所有权重参数初始化为零。虽然这是一种极端的方法，但在某些特定情况下可能有效。然而，[它容易导致网络的权重对称性](https://blog.csdn.net/sdu_hao/article/details/104719378#:~:text=%E5%8D%B3%E6%AF%8F%E5%B1%82%E7%9A%84%E5%90%84%E4%B8%AA%E8%8A%82%E7%82%B9%E5%85%B7%E6%9C%89%E5%AF%B9%E7%A7%B0%E6%80%A7%E3%80%82,%E8%BF%99%E6%A0%B7%E6%80%BB%E7%BB%93%E6%9D%A5%E7%9C%8B%EF%BC%9Aw%E5%88%9D%E5%A7%8B%E5%8C%96%E5%85%A8%E4%B8%BA0%EF%BC%8C%E5%BE%88%E5%8F%AF%E8%83%BD%E7%9B%B4%E6%8E%A5%E5%AF%BC%E8%87%B4%E6%A8%A1%E5%9E%8B%E5%A4%B1%E6%95%88%EF%BC%8C%E6%97%A0%E6%B3%95%E6%94%B6%E6%95%9B&%5D%E3%80%82%20%E5%9B%A0%E6%AD%A4%E5%8F%AF%E4%BB%A5%E5%AF%B9w%E5%88%9D%E5%A7%8B%E5%8C%96%E4%B8%BA%E9%9A%8F%E6%9C%BA%E5%80%BC%E8%A7%A3%E5%86%B3%EF%BC%88%E5%9C%A8cnn%E4%B8%AD%EF%BC%8Cw%E7%9A%84%E9%9A%8F%E6%9C%BA%E5%8C%96%EF%BC%8C%E4%B9%9F%E6%98%AF%E4%B8%BA%E4%BA%86%E4%BD%BF%E5%BE%97%E5%90%8C%E4%B8%80%E5%B1%82%E7%9A%84%E5%A4%9A%E4%B8%AAfilter%EF%BC%8C%E5%88%9D%E5%A7%8Bw%E4%B8%8D%E5%90%8C%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%AD%A6%E5%88%B0%E4%B8%8D%E5%90%8C%E7%9A%84%E7%89%B9%E5%BE%81%EF%BC%8C%E5%A6%82%E6%9E%9C%E9%83%BD%E6%98%AF0%E6%88%96%E6%9F%90%E4%B8%AA%E5%80%BC%EF%BC%8C%E7%94%B1%E4%BA%8E%E8%AE%A1%E7%AE%97%E6%96%B9%E5%BC%8F%E7%9B%B8%E5%90%8C%EF%BC%8C%E5%8F%AF%E8%83%BD%E8%BE%BE%E4%B8%8D%E5%88%B0%E5%AD%A6%E4%B9%A0%E4%B8%8D%E5%90%8C%E7%89%B9%E5%BE%81%E7%9A%84%E7%9B%AE%E7%9A%84%EF%BC%89)，因此通常不建议在深度神经网络中使用。

3. **Xavier初始化（Glorot初始化）**：适用于**S型激活函数（如sigmoid和tanh）**。它根据输入和输出的神经元数量来自动调整初始化权重的范围，以确保激活值在合适的范围内变化。这有助于避免梯度消失或梯度爆炸问题。

4. **He初始化**：适用于**ReLU（Rectified Linear Unit）激活函数**。它也根据输入和输出神经元的数量来调整初始化范围，但与Xavier初始化不同，它使用了更大的系数，以更好地适应ReLU的性质。

5. **自定义初始化**：有时，根据具体问题和网络结构，研究人员和工程师可能会设计自定义的初始化方法。这些方法可以根据领域知识或特定需求来初始化权重参数。

6. **预训练模型的初始化**：当使用预训练的神经网络模型时，通常会使用该模型在先前任务上训练的参数作为初始化。这些参数在大规模数据上进行了训练，通常能够提供很好的起始点，以便在新任务上微调模型。

选择哪种参数初始化方法通常取决于网络的架构、激活函数以及所解决的具体问题。不同的初始化方法可能会对模型的训练和性能产生显著影响，因此需要根据实验来选择最合适的初始化策略。
He初始化和Xavier初始化是两种常用的权重初始化方法，它们设计用于不同类型的激活函数，以促进网络的稳定训练和更好的性能。

>具体介绍Xavier初始化和He初始化

具体原文推导：[Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

1. **Xavier初始化（也称为Glorot初始化）：**

 	Xavier初始化适用于使用S型激活函数，如sigmoid和tanh。这种初始化方法的目标是使**每层输出的方差保持一致**，以确保信号在前向传播和反向传播过程中不会消失或爆炸。Xavier初始化的公式如下：

   对于一个全连接层，权重初始化为：
    $$W = \text{rand}(-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}})$$

   其中，n是前一层神经元的数量。对于卷积层，权重初始化也是类似的，只是n的计算稍有不同。

2. **He初始化：**

   He初始化适用于使用ReLU（Rectified Linear Unit）激活函数及其变种，如Leaky ReLU。ReLU在正区域有激活，而Xavier初始化可能不足以保持方差，因此需要更大的初始化范围。He初始化的公式如下：

   对于一个全连接层，权重初始化为：

   $$W = \text{rand}(-\sqrt{\frac{2}{n}}, \sqrt{\frac{2}{n}})$$

   其中，n是前一层神经元的数量。对于卷积层，权重初始化也是类似的，只是n的计算稍有不同。

### 从pytorch看内置初始化
默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵，这个范围是根据输入和输出维度计算出的。PyTorch的`nn.init`模块提供了多种预置初始化方法。


将所有权重参数初始化为标准差为0.01的高斯随机变量，且将偏置参数设置为0。
```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```
(tensor([-0.0214, -0.0015, -0.0100, -0.0058]), tensor(0.))

所有参数初始化为给定的常数，比如初始化为1。

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```
(tensor([1., 1., 1., 1.]), tensor(0.))

使用Xavier初始化方法初始化第一个神经网络层，然后将第三个神经网络层初始化为常量值42。

```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
### 自定义初始化

任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

1. 遍历神经网络的各个层，检查是否是nn.Linear层。
2. 对于nn.Linear层，使用均匀分布初始化权重，将权重限制在[-10, 10]之间。
3. 然后，通过将权重数据与大于等于5的绝对值元素进行相乘来将权重值修剪为零。`m.weight.data.abs() >= 5`其中的元素为True（真）如果对应的绝对值大于等于5，否则为False（假）。

`net[0].weight[:2]`表示选择神经网络的第0个层的权重的前两个元素。


### 共享参数——稠密层

```python
# 共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])

 - Q1：当参数绑定时，梯度会发生什么情况？
由于模型参数包含梯度，因此在反向传播期间第二个隐藏层（即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。

- Q2：构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个自定义的共享参数层
class SharedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SharedLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# 创建多层感知机模型
class MLPWithSharedLayer(nn.Module):
    def __init__(self):
        super(MLPWithSharedLayer, self).__init__()
        share=SharedLayer(10, 10)
        self.shared_layer1 = share  # 共享参数的层
        self.shared_layer2 = share  # 共享参数的层
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# 创建模型实例
model = MLPWithSharedLayer()

# 打印模型结构
print(model)

# 获取 shared_layer1 和 shared_layer2 的参数张量
params_shared_layer1 = list(model.shared_layer1.parameters())
params_shared_layer2 = list(model.shared_layer2.parameters())

# 检查权重和偏差是否相同
weights_shared_layer1 = params_shared_layer1[0]
biases_shared_layer1 = params_shared_layer1[1]

weights_shared_layer2 = params_shared_layer2[0]
biases_shared_layer2 = params_shared_layer2[1]

if torch.equal(weights_shared_layer1, weights_shared_layer2) and torch.equal(biases_shared_layer1, biases_shared_layer2):
    print("shared_layer1 和 shared_layer2 的参数相同。")
else:
    print("shared_layer1 和 shared_layer2 的参数不同。")

# 构造一个虚拟输入
input_data = torch.randn(1, 10)

# 前向传播
output = model(input_data)

# 定义一个损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 反向传播和参数更新
optimizer.zero_grad()
loss = criterion(output, torch.randn(1, 3))
loss.backward()
optimizer.step()

# 观察参数梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Layer: {name}, Gradient: {param.grad}")



```
输出的共享层一致。

- Q3：为什么共享参数是个好主意？

1. 减少参数量：共享参数可以减少模型的参数数量，从而降低模型的复杂度。这有助于减少过拟合的风险，特别是在数据集相对较小的情况下。

2. 提高模型泛化能力：共享参数可以使模型更具泛化能力，因为它可以从不同的输入示例中学到通用的特征表示。这意味着模型可以更好地适应新的、未见过的数据。


4. 处理变长输入：共享参数可以用于处理变长输入序列，如自然语言处理中的文本。这使得模型能够在不同长度的序列上进行预测，而无需针对每个长度都训练不同的模型。

5. 模型可解释性：共享参数可以帮助提高模型的可解释性，因为它可以学习一些通用的特征，这些特征对于任务的理解可能是有帮助的。这可以有助于深度学习模型更好地理解任务的本质。

卷积神经网络共享参数，也可以用于模型的可解释性。


## 延迟初始化

所忽略的：
* 我们定义了网络架构，但**没有指定输入维度**。
* 我们添加层时**没有指定前一层的输出维度**。
* 我们在初始化参数时，甚至没有足够的信息来确定模型应该包含**多少参数**。

解决：
*延后初始化*（defers initialization），即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

[实例](https://blog.csdn.net/weixin_43180762/article/details/124299823)

# 3.自定义层
## 不带参数的层
自定义层允许创建自己的神经网络层，以满足特定的任务或模型需求。

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```
该代码定义了这个自定义层的构造函数 __init__ 和前向传播函数 forward。在前向传播函数中，它执行了均值减法操作。

要使用这个自定义层，可以将其添加到自己的模型中。如下：

```python
import torch
import torch.nn.functional as F
from torch import nn

# 创建一个示例模型，包括CenteredLayer
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init()
        self.centered_layer = CenteredLayer()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, X):
        X = self.centered_layer(X)
        X = F.relu(self.fc1(X))
        return X

# 创建模型实例
model = MyModel()

# 使用模型进行前向传播
input_data = torch.randn(1, 10)  # 示例输入数据
output = model(input_data)

```

在这个示例中，首先创建了一个包括 CenteredLayer 自定义层的模型 MyModel，然后使用模型进行前向传播。在前向传播过程中，输入数据首先通过 CenteredLayer 层，然后通过一个全连接层。这个自定义层的作用是将输入数据中的均值中心化到零，然后传递给下一层进行处理。

## 带参数的层

```python
# 创建一个自定义的线性层 MyLinear
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 定义权重矩阵和偏置项作为模型参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        # 执行线性变换操作
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        # 应用ReLU激活函数
        return F.relu(linear)
```
这个自定义线性层实现了一个带有权重和偏置的线性变换，并在线性变换后应用了ReLU激活函数。

分析：

1. `__init__` 方法：在构造函数中，定义了该自定义层的输入单元数 `in_units` 和输出单元数 `units`，并使用`nn.Parameter`定义了权重矩阵 `weight` 和偏置向量 `bias`。这些参数将在模型训练过程中自动进行反向传播的梯度更新。

2. `forward` 方法：在前向传播方法中，执行了线性变换操作，使用权重矩阵 `self.weight` 和偏置向量 `self.bias` 对输入张量 `X` 进行线性组合，并将结果存储在 `linear` 变量中。然后，应用了ReLU激活函数 `F.relu`，将非线性性引入到层的输出中。


>设计一个接受输入并计算张量降维的层，它返回$y_k = \sum_{i, j} W_{ijk} x_i x_j$。

```python
import torch
from torch import nn

class CustomReductionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomReductionLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, in_features))

    def forward(self, x):
        # x的维度应该是 (batch_size, in_features)
        batch_size, in_features = x.size()

        # 对输入 x 进行逐元素相乘
        x_i = x.unsqueeze(1)  # 扩展维度 (batch_size, 1, in_features)
        x_j = x.unsqueeze(2)  # 扩展维度 (batch_size, in_features, 1)
        x_ij = torch.matmul(x_j, x_i)  # 计算 x_i * x_j (batch_size, in_features, in_features)

        # 使用权重矩阵与 x_ij 相乘，然后求和
        y = torch.sum(self.weight.view(-1, in_features, in_features) * x_ij, dim=(1, 2))

        return y

# 创建一个示例模型，包括自定义的降维层
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init()
        self.reduction_layer = CustomReductionLayer(3, 1)  # 输入特征数为3，输出特征数为1

    def forward(self, x):
        x = self.reduction_layer(x)
        return x

# 创建模型实例
model = MyModel()

# 使用模型进行前向传播
input_data = torch.randn(1, 3)  # 示例输入数据，维度为 (batch_size, in_features)
output = model(input_data)

```
代码解释：
当设计一个自定义降维层时，我们需要根据给定的公式计算输出。这个公式是:

$$y_k = \sum_{i, j} W_{ijk} x_i x_j$$

其中，$x$ 是输入张量，$W$ 是权重矩阵，$y_k$ 是输出张量中的一个元素。
解释代码中的各个部分：

1. `CustomReductionLayer` 类的构造函数：
   - 在构造函数中，我们定义了两个参数，`in_features` 和 `out_features`，分别表示输入特征的数量和输出特征的数量。
   - 我们使用 `nn.Parameter` 创建一个权重矩阵 `self.weight`，该权重矩阵的维度是 `(out_features, in_features, in_features)`。

2. `forward` 方法：
   - 在前向传播方法中，我们首先将输入 `x` 的维度扩展为 `(batch_size, 1, in_features)`，这是为了能够执行逐元素相乘。
   - 我们再将输入 `x` 的维度扩展为 `(batch_size, in_features, 1)`，以便与前面扩展的 `x` 相乘。
   - 接下来，我们使用 `torch.matmul` 计算 `x_j * x_i`，这将得到一个维度为 `(batch_size, in_features, in_features)` 的张量 `x_ij`，其中每个元素是 $x_i * x_j$ 的乘积。
   - 然后，我们将权重矩阵 `self.weight` 与 `x_ij` 相乘，通过逐元素相乘，得到一个 `(batch_size, in_features, in_features)` 的张量。
   - 最后，我们使用 `torch.sum` 对所有元素进行求和，得到输出 `y`。由于我们的权重矩阵被视为三维的，因此我们需要通过 `view` 将其重塑为 `(out_features, in_features, in_features)` 的形状以进行逐元素相乘。

3. 示例模型 `MyModel`：
   - 我们创建一个示例模型，其中包括一个 `CustomReductionLayer` 自定义降维层，该层具有输入特征数为3和输出特征数为1。
   - 在前向传播方法中，我们使用这个自定义降维层来处理输入数据 `x`，然后返回输出。

总之，这个自定义降维层实现了给定公式的计算过程，它将输入 `x` 映射到输出 `y`，其中 `y_k` 是公式中的一个元素。你可以将这个自定义层集成到更复杂的神经网络中，以满足具体的深度学习任务需求。


>设计一个返回输入数据的傅立叶系数前半部分的层。

```python
import torch
import torch.nn as nn

class FourierCoefficientsLayer(nn.Module):
    def __init__(self, num_coefficients):
        super(FourierCoefficientsLayer, self).__init__()
        self.num_coefficients = num_coefficients

    def forward(self, x):
        # Apply FFT to the input data
        output_fft = torch.fft.fft2(x, dim=(-2, -1))
        # Select the first half of the coefficients
        half_idx = self.num_coefficients // 2
        coefficients = output_fft[..., :half_idx]

        return coefficients

# 使用示例
num_coefficients = 128
input_dim = 256 # 输入数据的维度（这里使用了较小的维度作为示例）

# 创建模型
model = nn.Sequential(
    FourierCoefficientsLayer(num_coefficients),
)

# 生成输入数据（示例）
input_data = torch.randn(1, 3, input_dim, input_dim)

# 使用模型进行前向传播
output = model(input_data)

print(output.shape)  

```
# 读写文件
学习如何加载和存储权重向量和整个模型。
## 加载&保存张量
```python
import torch
from torch import nn
from torch.nn import functional as F
#张量
# 创建一个PyTorch张量 x
x = torch.arange(4)
# 使用 torch.save 将 x 保存到名为 'x-file' 的文件中
torch.save(x, 'x-file')
# 使用 torch.load 从文件中加载数据，并将其存储在 x2 变量中
x2 = torch.load('x-file')

#字典
# 创建一个包含两个 PyTorch 张量 x 和 y 的字典 mydict
mydict = {'x': x, 'y': y}
# 使用 torch.save 将 mydict 字典保存到名为 'mydict' 的文件中
torch.save(mydict, 'mydict')
# 使用 torch.load 从文件中加载数据，并将其存储在 mydict2 变量中
mydict2 = torch.load('mydict')

# mydict2 现在包含了与 mydict 相同的数据，即包含 x 和 y 两个张量的字典


```
## 加载&保存模型
深度学习框架可以帮助我们保存和加载整个神经网络，但有一个重要的细节：保存的是**神经网络的参数**，而不是整个神经网络的代码。

若有3个层的MLP，这个神经网络在训练过程中学到了如何处理数据，但它的架构（层的结构）是你自己设计的。由于神经网络的架构可以非常复杂，包含许多不同的层和连接，因此很难将整个神经网络的结构保存到磁盘上。

所以，为了保存和加载神经网络，通常我们需要分两步走。

- 首先，保存神经网络的参数，这些参数是训练过程中学到的权重和偏置值，它们定义了神经网络的行为。
- 然后，使用这个神经网络时，需要用代码来构建相同的神经网络结构（即相同的层和连接），并将之前保存的参数加载到这个结构中。这就是为什么需要单独指定神经网络的架构。

```python
import torch
import torch.nn as nn

# 创建一个自定义的多层感知器（MLP）神经网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个具有20个输入特征和256个隐藏单元的线性层
        self.hidden = nn.Linear(20, 256)
        # 定义一个具有256个输入特征和10个输出类别的线性层
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # 神经网络的前向传播函数
        # 使用ReLU激活函数作用在隐藏层上，然后将结果传递给输出层
        return self.output(F.relu(self.hidden(x)))

# 创建MLP模型的实例
net = MLP()

# 创建一个随机输入张量X，形状为(2, 20)
X = torch.randn(size=(2, 20))

# 使用MLP模型进行前向传播得到输出张量Y
Y = net(X)

# 保存模型参数到文件 'mlp.params'
torch.save(net.state_dict(), 'mlp.params')

# 创建另一个MLP模型的实例 'clone'
clone = MLP()

# 加载之前保存的模型参数到 'clone' 模型
clone.load_state_dict(torch.load('mlp.params'))

# 设置 'clone' 模型为评估模式
clone.eval()

# 使用克隆的模型进行前向传播得到输出张量 'Y_clone'
Y_clone = clone(X)

# 检查 'Y_clone' 是否与 'Y' 相等
# 这将返回一个包含布尔值的张量，用于比较两个张量的相等性
Y_clone == Y

```
 
 其中最重要的是：保存和加载模型。
 

```python
# 保存模型参数到文件 'mlp.params'
torch.save(net.state_dict(), 'mlp.params')

# 创建另一个MLP模型的实例 'clone'
clone = MLP()

# 加载之前保存的模型参数到 'clone' 模型
clone.load_state_dict(torch.load('mlp.params'))
```

>如果我想要冻结原始网络的部分层，以防止它们在训练过程中更新

要冻结原始网络的部分层，以防止它们在训练过程中更新，可以在新网络中使用 `requires_grad` 属性来控制参数是否可训练。通常，会将 `requires_grad` 属性设置为 `False`，以阻止参数的梯度更新。这可以通过以下方式实现：

1. 首先，创建新网络，并将原始网络的部分层添加到新网络。

2. 在新网络中，将要冻结的层的参数的 `requires_grad` 属性设置为 `False`。

3. 定义新网络的其余部分（如果有的话）以适应新的任务。

4. 在训练循环中，只优化新网络的参数，而不优化原始网络的部分层。

以下是一个示例代码片段，演示了如何冻结原始网络的部分层：

```python
import torch.nn as nn

class NewNetwork(nn.Module):
    def __init__(self, pretrained_network):
        super().__init()
        # 复用原始网络的前两层参数
        self.layer1 = pretrained_network.layer1
        self.layer2 = pretrained_network.layer2

        # 冻结原始网络的部分层，阻止其参数更新
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        # 在新网络中添加额外的层
        self.fc = nn.Linear(64, num_classes)  # 例如，添加一个全连接层

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x

# 创建原始网络
pretrained_network = OriginalNetwork()

# 创建新网络，传递原始网络作为参数
new_network = NewNetwork(pretrained_network)

# 定义损失函数和优化器，只优化新网络的参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(new_network.parameters(), lr=0.1)

# 在训练循环中，只更新新网络的参数
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = new_network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

在上述示例中，`requires_grad` 属性被设置为 `False`，以防止原始网络的层的参数更新。然后，只有新网络的参数会在训练过程中得到更新。这可以在新任务上训练新网络，而不会影响原始网络的参数。



 >假设我们只想复用网络的一部分，以将其合并到不同的网络架构中。比如想在一个新的网络中使用之前网络的前两层，该怎么做？


   - 创建一个新的网络，其中包含想要重用的部分（前两层）。这部分通常会在新网络的构造函数中定义。
   - 加载原始网络的参数（前两层对应的参数），并将它们分配给新网络的相应层。
   - 在新网络的构造函数中定义新的层，以匹配新网络的架构和任务。
   - 可以选择是否冻结原始网络的部分层，以防止它们在训练过程中更新。

   下面是一个示例代码片段

   ```python
   class NewNetwork(nn.Module):
       def __init__(self, pretrained_network):
           super().__init()
           # 复用原始网络的前两层参数
           self.layer1 = pretrained_network.layer1
           self.layer2 = pretrained_network.layer2
           # 在新网络中添加额外的层
           self.fc = nn.Linear(64, num_classes)  # 例如，添加一个全连接层
   
       def forward(self, x):
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.fc(x)
           return x
   
   # 创建原始网络
   pretrained_network = OriginalNetwork()
   # 创建新网络，传递原始网络作为参数
   new_network = NewNetwork(pretrained_network)
   ```

>如何同时保存和加载模型的架构和参数：

   ```python
   # 保存模型的架构和参数
   torch.save(model, 'model.pth')
   # 加载模型的架构和参数
   loaded_model = torch.load('model.pth')
   ```

   这将保存整个模型，包括架构和参数，然后可以加载它以重新创建完整的模型。要成功加载模型，确保在加载时定义与保存时相同的模型类，以便正确重建模型架构。
# GPU
对于pytorch上面的CPU和GPU：
在PyTorch中，CPU和GPU可以用`torch.device('cpu')`和`torch.device('cuda')`表示。
`cpu`设备意味着所有物理CPU和内存——PyTorch的计算将尝试使用所有CPU核心。
`gpu`设备只代表一个卡和相应的显存。如果有多个GPU，我们使用`torch.device(f'cuda:{i}')`来表示第$i$块GPU（$i$从0开始）。

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```
(device(type='cpu'), device(type='cuda'), device(type='cuda', index=1))

我们可以(**查询可用gpu的数量。**)
```python
torch.cuda.device_count()
```
## 创建张量
默认是在cpu上面创建张量

```python
x = torch.tensor([1, 2, 3])
x.device
```
device(type='cpu')


加上：device=try_gpu()，才采用gpu。

```python
X = torch.ones(2, 3, device=try_gpu())
X
```

tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')


## 神经网络和GPU
神经网络和GPU：
神经网络模型可以指定设备。	下面的代码将模型参数放在GPU上。

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```
## 比较GPU和CPU
> 尝试一个计算量更大的任务，比如大矩阵的乘法，看看CPU和GPU之间的速度差异。再试一个计算量很小的任务呢？


首先，从大矩阵乘法开始。

```python
import numpy as np
import time
import cupy as cp  # 导入 CuPy 库，用于 GPU 计算

# 定义矩阵的大小
matrix_size = 1000

# 创建两个随机矩阵
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

# 在CPU上执行矩阵乘法并测量时间
start_time = time.time()
result_cpu = np.dot(matrix_a, matrix_b)
end_time = time.time()
cpu_time = end_time - start_time

# 在GPU上执行矩阵乘法并测量时间
matrix_a_gpu = cp.asarray(matrix_a)
matrix_b_gpu = cp.asarray(matrix_b)
start_time = time.time()
result_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)
cp.cuda.Stream.null.synchronize()  # 确保 GPU 计算完成
end_time = time.time()
gpu_time = end_time - start_time

print(f"CPU 矩阵乘法时间：{cpu_time} 秒")
print(f"GPU 矩阵乘法时间：{gpu_time} 秒")
```

上述代码中，生成两个大小为1000x1000的随机矩阵，然后分别在CPU和GPU上执行矩阵乘法，测量所需的时间。
CPU 矩阵乘法时间：0.06453061103820801 秒
GPU 矩阵乘法时间：0.04992389678955078 秒

接下来，对于小计算任务，我们可以使用一个简单的求和来演示速度差异：

```python
import numpy as np
import time
import cupy as cp  # 导入 CuPy 库，用于 GPU 计算

# 定义一个小计算任务
n = 10000
data_cpu = np.random.rand(n)
data_gpu = cp.random.rand(n)

# 在CPU上执行小计算任务并测量时间
start_time = time.time()
result_cpu = sum(data_cpu)
end_time = time.time()
cpu_time = end_time - start_time

# 在GPU上执行小计算任务并测量时间
start_time = time.time()
result_gpu = cp.sum(data_gpu)
cp.cuda.Stream.null.synchronize()  # 确保 GPU 计算完成
end_time = time.time()
gpu_time = end_time - start_time

print(f"CPU 小计算任务时间：{cpu_time} 秒")
print(f"GPU 小计算任务时间：{gpu_time} 秒")
```
CPU 小计算任务时间：0.0009222030639648438 秒
GPU 小计算任务时间：0.04811906814575195 秒
这个示例演示了在小计算任务上，CPU通常会更快，因为在这种情况下，GPU的并行计算能力不会得到充分利用。

 - Q1为：CPU和GPU的优劣势：

1. 并行计算能力：GPU是专门设计用于**并行计算的处理器**，它们具有大量的小处理单元，可以同时执行多个任务。因此**在需要大量并行计算的任务（如矩阵乘法）中，GPU通常比CPU更快，因为它们可以同时处理多个元素。**

2. 计算密集型任务：对于需要大量数学运算的任务，如矩阵乘法，GPU通常会更快，因为它们在执行这些计算时具有更高的吞吐量。GPU的处理器数量较多，适合并行计算，而且通常拥有更大的内存带宽。

3. 小规模任务和控制流：CPU在处理小规模任务和需要频繁的控制流改变时表现更好。这是因为**CPU具有更强的单线程性能和更复杂的控制单元，适合执行复杂的逻辑和串行计算**。

4. 内存访问：GPU内存通常较大，但访问延迟较高。对于某些任务，特别是那些具有不规则的内存访问模式的任务，CPU的高速缓存层次结构可能更有效。

对于计算密集型、高度并行的任务，GPU通常更适合，但对于小规模任务或需要频繁的控制流的任务，CPU可能更有效。

