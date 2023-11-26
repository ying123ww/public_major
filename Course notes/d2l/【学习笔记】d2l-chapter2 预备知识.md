# 前言
本章主要作为学习+查询工具，之后有什么不太记得的语句，可以返回来用ctrl+F来查询。

# 数据操作


$n$维数组，也称为*张量*（tensor）。
注意：*张量类*（在MXNet中为`ndarray`，在PyTorch和TensorFlow中为`Tensor`）都与Numpy的`ndarray`类似。
但深度学习框架又比Numpy的`ndarray`多一些重要功能：
1. GPU很好地支持加速计算，而NumPy仅支持CPU计算；
2. 张量类支持自动微分。

## 入门


```python
import torch
```


具有一个轴的张量对应数学上的*向量*（vector）；
具有两个轴的张量对应数学上的*矩阵*（matrix）；



首先，使用 `arange` 创建一个行向量 `x`。这个行向量包含以0开始的前12个整数，它们默认创建为整数。也可指定创建类型为浮点数。张量中的每个值都称为张量的 *元素*（element）。例如，张量 `x` 中有 12 个元素。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。

创建行向量：
```python
x = torch.arange(12)
x
```

tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])


向量形状：
```python
x.shape
```

torch.Size([12])


张量 x 的元素总数：
```python
x.numel()
```



改变形状：
```python
X = x.reshape(3, 4) #例如，可以把张量`x`从形状为（12,）的行向量转换为形状为（3,4）的矩阵。
X
```
\tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])

`-1`来调用此自动计算出维度的功能。可以用`x.reshape(-1,4)`或`x.reshape(3,-1)`来取代`x.reshape(3,4)`。


全零张量，形状（2,3,4）：
```python
torch.zeros((2, 3, 4))
```




tensor([[[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
    [[0., 0., 0., 0.],
		 [0., 0., 0., 0.],
		 [0., 0., 0., 0.]]])



全1张量：
```python
torch.ones((2, 3, 4))
```


随机采样（服从正态分布）
```python
torch.randn(3, 4)
```

tensor([[ 0.7141,  0.8175,  0.6157, -0.4534],
            [-0.9941, -0.8847, -1.2346, -0.7467],
            [ 0.5641,  0.9925, -0.1348,  0.4283]])





列表赋值：
通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。
在这里，最外层的列表对应于轴0，内层的列表对应于轴1。
```python
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```
tensor([[2, 1, 4, 3],
            [1, 2, 3, 4],
            [4, 3, 2, 1]])



## 运算符


**常见的标准算术运算符（`+`、`-`、`*`、`/`和`**`）都可以被升级为按元素运算**。


```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

(tensor([ 3.,  4.,  6., 10.]),
     tensor([-1.,  0.,  2.,  6.]),
     tensor([ 2.,  4.,  8., 16.]),
     tensor([0.5000, 1.0000, 2.0000, 4.0000]),
     tensor([ 1.,  4., 16., 64.]))

求幂：
```python
torch.exp(x)
```

tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])





多个张量*连结*（concatenate）在一起：

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))#输出张量的轴-0长度（$6$）是两个输入张量轴-0长度的总和（$3 + 3$）
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])#输出张量的轴-1长度（$8$）是两个输入张量轴-1长度的总和（$4 + 4$）
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```
(tensor([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.],
             [ 2.,  1.,  4.,  3.],
             [ 1.,  2.,  3.,  4.],
             [ 4.,  3.,  2.,  1.]]),
     tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
             [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
             [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))




求和：
```python
X.sum()
```
tensor(66.)

## 广播机制

即使形状不同，通过调用*广播机制*（broadcasting mechanism）来执行按元素操作。
这种机制的工作方式如下：
1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
2. 对生成的数组执行按元素操作。

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

(tensor([[0],
             [1],
             [2]]),
     tensor([ [0, 1]]))




## 索引和切片

索引：
可以用`[-1]`选择最后一个元素，可以用`[1:3]`选择第二个和第三个元素：
```python
X[-1], X[1:3]
```
(tensor([ 8.,  9., 10., 11.]),
     tensor([[ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]]))

查找写入：
```python
X[1, 2] = 9
X
```
tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  9.,  7.],
            [ 8.,  9., 10., 11.]])



为多个元素赋值相同的值
```python
X[0:2, :] = 12
X
```

tensor([[12., 12., 12., 12.],
            [12., 12., 12., 12.],
            [ 8.,  9., 10., 11.]])



## 节省内存

[**运行一些操作可能会导致为新结果分配内存**]。
例如，如果我们用`Y = X + Y`，我们将取消引用`Y`指向的张量，而是指向新分配的内存处的张量。


运行`Y = Y + X`后，我们会发现`id(Y)`指向另一个位置。
这是因为Python首先计算`Y + X`，为结果分配新的内存，然后使`Y`指向内存中的这个新位置。

```python
before = id(Y)
Y = Y + X
id(Y) == before
```

False

这可能是不可取的，原因有两个：

1. 首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新；
2. 如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如`Y[:] = <expression>`。
```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

id(Z): 1546620366576
id(Z): 1546620366576
    

如果在后续计算中没有重复使用`X`，我们也可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。

```python
before = id(X)
X += Y
id(X) == before
```

 True

## 转换为其他Python对象

转换为NumPy张量（`ndarray`）
```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```
(numpy.ndarray, torch.Tensor)

转换为标量：
```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

(tensor([3.5000]), 3.5, 3.5, 3)



# 数据预处理（pandas）
`
## 读取数据集

按行写入CSV文件：
```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

读取csv文件：
```python
、
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

       NumRooms Alley   Price
    0       NaN  Pave  127500
    1       2.0   NaN  106000
    2       4.0   NaN  178100
    3       NaN   NaN  140000
    

## 处理缺失值

插值法：
通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`，
其中前者为`data`的前两列，而后者为`data`的最后一列。
对于`inputs`中缺少的数值，我们用同一列的均值替换“NaN”项。

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

       NumRooms Alley
    0       3.0  Pave
    1       2.0   NaN
    2       4.0   NaN
    3       3.0   NaN
    

**对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**
由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，
`pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

       NumRooms  Alley_Pave  Alley_nan
    0       3.0           1          0
    1       2.0           0          1
    2       4.0           0          1
    3       3.0           0          1
    

## 转换为张量格式

数值类型——换为张量格式
```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
```


(tensor([[3., 1., 0.],
             [2., 0., 1.],
             [4., 0., 1.],
             [3., 0., 1.]], dtype=torch.float64),
     tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))









# 微积分
`

## 导数


假设我们有一个函数$f: \mathbb{R} \rightarrow \mathbb{R}$，其输入和输出都是标量。
**如果$f$的*导数*存在，这个极限被定义为**

**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**
`

如果$f'(a)$存在，则称$f$在$a$处是*可微*（differentiable）的。



## 偏导数

将微分的思想推广到多元函数（multivariate function）上。

设$y = f(x_1, x_2, \ldots, x_n)$是一个具有$n$个变量的函数。
$y$关于第$i$个参数$x_i$的*偏导数*（partial derivative）为：

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

为了计算$\frac{\partial y}{\partial x_i}$，我们可以简单地将$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$看作常数，并计算$y$关于$x_i$的导数。
对于偏导数的表示，以下是等价的：

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 梯度


我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。
具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是
一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。
函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。

假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用以下规则:

* 对于所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。


## 链式法则


在深度学习中，多元函数通常是*复合*（composite）的，链式法则可以被用来微分复合函数。

让我们先考虑单变量函数。假设函数$y=f(u)$和$u=g(x)$都是可微的，根据链式法则：

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

现在考虑一个更一般的场景，即函数具有任意数量的变量的情况。
假设可微分函数$y$有变量$u_1, u_2, \ldots, u_m$，其中每个可微分函数$u_i$都有变量$x_1, x_2, \ldots, x_n$。
注意，$y$是$x_1, x_2， \ldots, x_n$的函数。
对于任意$i = 1, 2, \ldots, n$，链式法则给出：

$$\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}$$


# 自动微分


深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。
根据设计好的模型，系统会构建一个*计算图*（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。

## 例子

**假设我们想对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导**。
首先，我们创建变量`x`并为其分配一个初始值。

```python
import torch
x = torch.arange(4.0)
x
```

```python
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
```

(**现在计算$y$。**)

```python
y = 2 * torch.dot(x, x)
```

`x`是一个长度为4的向量，计算`x`和`x`的点积，得到了我们赋值给`y`的标量输出。
接下来，**通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度**，并打印这些梯度。

```python
y.backward()
x.grad
```

tensor([ 0.,  4.,  8., 12.])

函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于$\mathbf{x}$的梯度应为$4\mathbf{x}$。
验证：
```python
x.grad == 4 * x
```

tensor([True, True, True, True])



**现在计算`x`的另一个函数。**

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```
tensor([1., 1., 1., 1.])



## 非标量变量的反向传播

```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

tensor([0., 2., 4., 6.])



## 分离计算

比如想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数，
并且只考虑到`x`在`y`被计算后发挥的作用。这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值，但丢弃计算图中如何计算`y`的任何信息。换句话说，梯度不会向后流经`u`到`x`。


```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

tensor([True, True, True, True])



由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播，
得到`y=x*x`关于的`x`的导数，即`2*x`。
```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

```

tensor([True, True, True, True])



## Python控制流的梯度计算

使用自动微分的一个好处是：
**即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度**。
在下面的代码中，`while`循环的迭代次数和`if`语句的结果都取决于输入`a`的值。

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

计算梯度
```python
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

## 练习

>为什么计算二阶导数比一阶导数的开销要更大？

1. 计算二阶导数比一阶导数的开销更大，因为二阶导数涉及到更多的计算和内存消耗。一阶导数是函数的斜率，而二阶导数是一阶导数的导数，也就是函数的曲率。计算二阶导数需要计算函数的一阶导数，然后再次计算这些一阶导数的导数，因此它涉及到两个步骤。在计算机中，这通常需要更多的内存和计算时间。

>在运行反向传播函数之后，立即再次运行它，看看会发生什么。


如果在运行反向传播函数之后立即再次运行它，通常会发生错误。反向传播是基于计算图的，计算图在每次运行前向传播时被构建，然后在反向传播时用于计算梯度。一旦反向传播函数运行完毕，计算图就被销毁，因此无法再次运行它来计算梯度。如果你希望多次计算梯度，通常需要使用retain_graph=True选项来告诉PyTorch保留计算图。


```python
import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 创建一个形状为(3,)的随机向量a
a = torch.randn(3, requires_grad=True)

d = f(a)
d.backward()

#会报错
#你遇到的错误信息 "RuntimeError: grad can be implicitly created only for scalar outputs" 意味着在使用 backward() 方法计算梯度时，PyTorch 只能为标量输出（scalar outputs）隐式地创建梯度，而不能为非标量输出创建梯度。

```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    d:\master\study_task\first_level(8.19-11)\d2l-zh\pytorch\chapter_preliminaries\autograd.ipynb 单元格 27 line 1
         <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_preliminaries/autograd.ipynb#X42sZmlsZQ%3D%3D?line=13'>14</a> a = torch.randn(3, requires_grad=True)
         <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_preliminaries/autograd.ipynb#X42sZmlsZQ%3D%3D?line=15'>16</a> d = f(a)
    ---> <a href='vscode-notebook-cell:/d%3A/master/study_task/first_level%288.19-11%29/d2l-zh/pytorch/chapter_preliminaries/autograd.ipynb#X42sZmlsZQ%3D%3D?line=16'>17</a> d.backward()
    

    File c:\Users\ying\.conda\envs\d2l\lib\site-packages\torch\_tensor.py:487, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        477 if has_torch_function_unary(self):
        478     return handle_torch_function(
        479         Tensor.backward,
        480         (self,),
       (...)
        485         inputs=inputs,
        486     )
    --> 487 torch.autograd.backward(
        488     self, gradient, retain_graph, create_graph, inputs=inputs
        489 )
    

    File c:\Users\ying\.conda\envs\d2l\lib\site-packages\torch\autograd\__init__.py:193, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        189 inputs = (inputs,) if isinstance(inputs, torch.Tensor) else \
        190     tuple(inputs) if inputs is not None else tuple()
        192 grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    --> 193 grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
        194 if retain_graph is None:
        195     retain_graph = create_graph
    

    File c:\Users\ying\.conda\envs\d2l\lib\site-packages\torch\autograd\__init__.py:88, in _make_grads(outputs, grads, is_grads_batched)
         86 if out.requires_grad:
         87     if out.numel() != 1:
    ---> 88         raise RuntimeError("grad can be implicitly created only for scalar outputs")
         89     new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
         90 else:
    

    RuntimeError: grad can be implicitly created only for scalar outputs



```python
import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f(x) = sin(x)
def f(x):
    return np.sin(x)

# 定义计算导数的函数
def df(x, h=1e-5): #h无限逼近0
    return (f(x + h) - f(x - h)) / (2 * h)

# 创建 x 值范围
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = f(x)
y_prime_approx = df(x)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="f(x) = sin(x)")
plt.plot(x, y_prime_approx, label="Approximate df(x)/dx", linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Plot of f(x) and Approximate df(x)/dx")
plt.grid(True)
plt.show()

```


