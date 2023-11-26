# 前言
1. 门控循环单元（GRU）：
   - 引入了门控循环单元（GRU），一种用于解决数值不稳定性问题的RNN变体。
   - 解释了GRU的结构和工作原理，包括更新门和重置门。
   - 讨论了如何在深度学习框架中实现GRU。

2. 长短期记忆网络（LSTM）：
   - 介绍了长短期记忆网络（LSTM），另一种用于处理数值不稳定性问题的RNN变体。
   - 解释了LSTM的结构和工作原理，包括输入门、遗忘门和输出门。
   - 讨论了如何在深度学习框架中实现LSTM。

3. 深层循环神经网络：
   - 探讨了如何构建深层循环神经网络，其中RNN层堆叠在一起以增加模型的表示能力。
   - 讨论了深层RNN的梯度传播和训练技巧。

4. 双向循环神经网络：
   - 介绍了双向循环神经网络，一种能够利用上下文信息的RNN结构。
   - 解释了双向RNN的结构和工作原理，包括前向和后向传播。
   - 讨论了如何在序列学习任务中应用双向RNN。

5. 应用案例：语言建模：
   - 使用语言建模问题作为案例研究，演示如何应用高级RNN模型来处理序列数据。
   - 引入了编码器-解码器架构和束搜索，用于机器翻译等序列生成任务。
# GRU（门控循环网络）

GRU是LSTM的稍微简化的变体，通常能够提供同等效果，并且计算速度上更加快。

![[Pasted image 20231106093733.png]]

门控循环单元和普通循环神经网络之间的关键区别在于：前者**支持隐状态的门控**。模型有专门的机制来确定何时更新隐状态，以及何时重置隐状态。
## 重置门和更新门


*重置门*（reset gate）和*更新门*（update gate）
相同点：
- $(0, 1)$区间中的向量。
- 输入是由当前时间步的输入和前一时间步的隐状态。
- 输出是由使用sigmoid激活函数的两个全连接层给出。
![[Pasted image 20231106094433.png]]
于是可得公式：
对于给定的时间步$t$，假设输入是一个小批量$\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本个数$n$，输入个数$d$），
上一个时间步的隐状态是$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$（隐藏单元个数$h$）。那么，重置门$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和更新门$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$的计算如下所示：

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

其中$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$是偏置参数。



不同点：
- 重置门$R$ 允许我们控制“可能还想记住”的过去状态的数量。
- 更新门$Z$ 将允许我们控制新状态中有多少个是旧状态的副本。

## 候选隐状态——运用重置门

重置门$\mathbf{R}_t$ 与 常规隐状态更新机制 两者集成，得到在时间步$t$的*候选隐状态*（candidate hidden state）$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$。

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
其中$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$是偏置项，符号$\odot$是Hadamard积（按元素乘积）运算符。在这里，我们使用tanh非线性激活函数来确保候选隐状态中的值保持在区间$(-1, 1)$中。

这时候重新回到重置门的含义：重置门$R$ 允许我们控制“可能还想记住”的过去状态的数量。$\mathbf{R}_t \odot \mathbf{H}_{t-1}$ 就可以看到重置门$R$ 对于前一个隐状态 $H_{t-1}$ 的调节作用。
- 如果$R_t$ 接近 $1$ ,则恢复普通的循环神经网络。
- 如果$R_t$ 接近 $0$ ,则是MLP。

## 隐状态——运用更新门

新的隐状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$在多大程度上来自旧的状态$\mathbf{H}_{t-1}$和新的候选状态$\tilde{\mathbf{H}}_t$。
这就得出了门控循环单元的最终更新公式：

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

这时候重新回到更新门1的含义：更新门$Z$ 将允许我们控制新状态中有多少个是旧状态的副本。
- 如果更新门$\mathbf{Z}_t$接近$1$时，模型就倾向只保留旧状态。此时，来自$\mathbf{X}_t$的信息基本上被忽略，从而跳过了依赖链条中的时间步$t$。
- 如果更新门$\mathbf{Z}_t$接近$0$时，新的隐状态$\mathbf{H}_t$就会接近候选隐状态$\tilde{\mathbf{H}}_t$。

## 从0实现

1. 读取数据集，和上一章训练的数据集一样。

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

2. 初始化模型参数

```python
def get_params(vocab_size, num_hiddens, device):
    # 定义输入和输出的维度，通常等于词汇表的大小
    num_inputs = num_outputs = vocab_size

    # 辅助函数：生成服从正态分布的随机张量
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 辅助函数：生成门控循环单元（GRU）相关参数
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化更新门参数（W_xz, W_hz, b_z）
    W_xz, W_hz, b_z = three()
    # 初始化重置门参数（W_xr, W_hr, b_r）
    W_xr, W_hr, b_r = three()
    # 初始化候选隐状态参数（W_xh, W_hh, b_h）
    W_xh, W_hh, b_h = three()
    # 初始化输出层参数（W_hq, b_q）
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 将所有参数放入列表
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    
    # 设置所有参数的梯度属性为True，以便进行反向传播
    for param in params:
        param.requires_grad_(True)

    # 返回包含所有模型参数的列表
    return params

```

3. 定义模型初始化函数

返回一个形状为（批量大小，隐藏单元个数）的张量，张量的值全部为零。
```python
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

4. 定义门控循环单元模型

与公式一一对应。
```python
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

5. 训练与预测

```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 简洁实现

```python
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 总结
* 门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。
* 重置门有助于捕获序列中的短期依赖关系。
* 更新门有助于捕获序列中的长期依赖关系。
* 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。


# LSTM（长短期记忆网络）

LSTM借鉴计算机内的逻辑门，引入记忆元（memory cell），或称单元（cell）。（记忆元或许是隐状态的特殊类型）。设计目的是为了记录附加信息。
为了控制记忆元，于是设计：
1. *输出门*（output gate）用来从单元中输出条目。
2. *输入门*（input gate）用来决定何时将数据读入单元。
3. *遗忘门*（forget gate）重置单元的内容。
![[Pasted image 20231106102258.png]]

## 输入门、遗忘门、输出门

input：当前时间步的输入和前一个时间步的隐状态。
激活函数：sigmoid

假设有$h$个隐藏单元，批量大小为$n$，输入数为$d$。输入为$\mathbf{X}_t \in \mathbb{R}^{n \times d}$，前一时间步的隐状态为$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$。输入门是$\mathbf{I}_t \in \mathbb{R}^{n \times h}$，遗忘门是$\mathbf{F}_t \in \mathbb{R}^{n \times h}$，输出门是$\mathbf{O}_t \in \mathbb{R}^{n \times h}$。
它们的计算方法如下：

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

其中$\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$是偏置参数。

## 候选记忆元
与上面门不同的是，激活函数为tanh。则函数范围为(-1,1).
$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

总结为图上：
![[Pasted image 20231106102806.png]]

## 记忆元——运用输入门和遗忘门

- 输入门$\mathbf{I}_t$控制采用多少来自$\tilde{\mathbf{C}}_t$的新数据，
- 遗忘门$\mathbf{F}_t$控制保留多少过去的记忆元$\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$的内容。

于是得到公式：
$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

如果遗忘门始终为$1$且输入门始终为$0$，则过去的记忆元$\mathbf{C}_{t-1}$ 将随时间被保存并传递到当前时间步。
引入这种设计是为了缓解梯度消失问题，能更好地捕获序列中的长距离依赖关系。

## 隐状态——运用输出门

隐状态仅仅是**记忆元的$\tanh$的门控版本**。这就确保了$\mathbf{H}_t$的值始终在区间$(-1, 1)$内：

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

- 输出门接近$1$，有效地将所有记忆信息传递给预测部分。
- 输出门接近$0$，只保留记忆元内的所有信息，而不需要更新隐状态。

![[Pasted image 20231106103518.png]]

## 从0实现

1. 加载数据集
```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

2. 初始化模型参数
```python
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

```



3. 初始化函数
需要额外返回一个记忆元（memory cell）
```python
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

4. 实际模型
提供三个门和一个额外的记忆元。请注意，只有隐状态才会传递到输出层，而记忆元$\mathbf{C}_t$不直接参与输出计算。
```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```


5. 训练
```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```
## 简洁实现

```python
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

但是长短期记忆网络是典型的具有重要状态控制的隐变量自回归模型。
多年来已经提出了其许多变体，例如，多层、残差连接、不同类型的正则化。
然而，由于序列的长距离依赖性，训练LSTM和GRU的成本是相当高的。在后面的内容中，我们将讲述更高级的替代模型，如Transformer。

## 总结

* 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。
* 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。
* 长短期记忆网络可以缓解梯度消失和梯度爆炸。


# 深度循环神经网络

将单层变成多层的问题：如何添加更多层，在哪里添加额外的非线性层。
解决方式：将多层循环神经网络堆叠在一起。
于是产生了*堆叠循环神经网络(Stacked Recurrent Neural Network,SRNN)*，即把多个循环网络堆叠起来。
![[Pasted image 20231106105410.png]]


## 隐状态
假设在时间步$t$有一个小批量的输入数据$\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本数：$n$，每个样本中的输入数：$d$）。
将$l^\mathrm{th}$隐藏层（$l=1,\ldots,L$）的隐状态设为$\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$（隐藏单元数：$h$），
输出层变量设为$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（输出数：$q$）。
设置$\mathbf{H}_t^{(0)} = \mathbf{X}_t$，第$l$个隐藏层的隐状态使用激活函数$\phi_l$，则：

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
$H$ 上标表示层数，下标表示时间步。

参数：权重$\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$，$\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$和偏置$\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$都是第$l$个隐藏层的模型参数。

## 输出
最后，输出层的计算仅基于第$l$个隐藏层最终的隐状态：

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

参数：权重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$都是输出层的模型参数。
超参数：隐藏层数目$L$和隐藏单元数目$h$。

如果用GRU和LSTM的隐状态更新公式，则可以替换称深度GRU or 深度LSTM。

## 简洁实现
以长短期记忆网络模型为例，该代码与之前代码非常相似，实际上唯一的区别是我们指定了层的数量，而不是使用单一层这个默认值。

因为我们有不同的词元，所以输入和输出都选择相同数量，即`vocab_size`。隐藏单元的数量仍然是$256$。唯一的区别是，我们现在(**通过`num_layers`的值来设定隐藏层数**)。
```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
# 这里改变了层数
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
```
![[Pasted image 20231106110201.png]]

# 双向循环神经网络

由于现实生活中词语的填空，导致这个位置可能不止需要上文的语义信息，还需要下文的语义信息。

## 隐马尔可夫中的动态规划（未完成
>什么是隐马尔可夫模型？

隐马尔可夫模型（Hidden Markov Model；缩写：HMM）是统计模型，用来描述一个含有隐含未知参数的马尔可夫过程。其难点是从可观察的参数中确定该过程的隐含参数。隐马尔可夫模型中，**状态并不是直接可见的，但受状态影响的某些变量则是可见的**。每一个状态在可能输出的符号上都有一概率分布。

HMM 包含两个关键部分：

- 隐藏状态（Hidden States）：这些状态在模型内部存在，但不能直接观察到。隐藏状态形成一个状态链，每个状态在给定时间步上都有一个概率分布，用于表示下一个隐藏状态是什么。
- 观察序列（Observation Sequence）：这些是在每个时间步上可观察到的数据，通常是离散或连续的特征向量。观察序列的生成是由隐藏状态序列控制的。

HMM 的基本假设是马尔可夫性质，即当前的隐藏状态只依赖于前一个隐藏状态，而不依赖于更早的状态。此外，HMM 假设生成观察序列的过程也是马尔可夫性质的，即当前观察只依赖于当前隐藏状态。

![[Pasted image 20231106132045.png]]

上图x为观察序列，h为隐变量。
HMM有三个典型(canonical)问题:
* 预测(filter)：已知模型参数和某一特定输出序列，求最后时刻各个隐含状态的概率分布，即求 $P(x(t)\mid y(1),\ldots,y(t)).$通常使用前向算法解决。
* 平滑(smoothing)：已知模型参数和某一特定输出序列，求中间时刻各个隐含状态的概率分布，即求 $P(x(k)\mid y(1),\ldots,y(t)),k<t$。通常使用前向-后向算法解决。
* 解码(most likely explanation)：已知模型参数，寻找最可能的能产生某一特定输出序列的隐含状态的序列，即求$P([x(1)\ldots x(t)]|[y(1)\ldots,y(t)])$ <math> P( [x(1) \dots x(t)]  |  [y(1) \dots ,y(t)] ) </math>。通常使用Viterbi算法解决。



我们现在要做的是第二个问题：平滑。

因此，对于有$T$个观测值的序列，在观测状态和隐状态上具有以下联合概率分布：

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$


假设我们观测到所有的$x_i$，除了$x_j$，并且我们的目标是计算$P(x_j \mid x_{-j})$，其中$x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$。
由于$P(x_j \mid x_{-j})$中没有隐变量，因此我们考虑对$h_1, \ldots, h_T$选择构成的所有可能的组合进行求和。
如果任何$h_i$可以接受$k$个不同的值（有限的状态数），这意味着我们需要对$k^T$个项求和，
有个巧妙的解决方案：*动态规划*（dynamic programming）。

？？？？？这一块不是很会orz，先跳过吧，学也学不会。

## 双向模型

![[Pasted image 20231106144140.png]]
上图是双向循环神经网络框架图。而且算是一层。

双向循环神经网络RNN添加了反向传递信息的隐藏层。使得模型具有前瞻能力。

定义：
对于任意时间步$t$，给定一个小批量的输入数据$\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本数$n$，每个示例中的输入数$d$），并且令隐藏层激活函数为$\phi$。在双向架构中，我们设该时间步的前向和反向隐状态分别为$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$和
$\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$，其中$h$是隐藏单元的数目。
前向和反向隐状态的更新如下：
$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

模型参数：权重$\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$和偏置$\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h}, \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$

**前向隐状态$\overrightarrow{\mathbf{H}}_t$和反向隐状态$\overleftarrow{\mathbf{H}}_t$连接起来，得到隐状态$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$**
（如果具有多个隐藏层，则该信息作为输入传递到下一个双向层）

最后，输出层计算得到的输出为$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（$q$是输出单元的数目）：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

输出层的模型参数：权重矩阵$\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$

## 应用领域
由于双向循环神经网络的特性就是使用来自过去和未来的观测信息来预测当前的观测。这时候就要求，训练的时候就需要上下文信息，那么测试的时候提供的数据也要有上下文信息，这样预测出来的结果才会准确。

## 计算成本

由于是双链，所以速度非常满。其主要原因是网络的前向传播需要在双向层中进行前向和后向递归，并且网络的反向传播还依赖于前向传播的结果。因此，梯度求解将有一个非常长的链。

# 机器翻译与数据集

机器翻译(machine translation)是<u>将输入序列转化成输出学列的序列转化模型(sequence transduction model)</u>的核心问题。

机器翻译有两大类，一个是统计机器翻译，一个是神经机器翻译。本书关注神经机器翻译（强调端到端的学习）

数据集：源语言和目标语言的文本序列对。
	Go.	Va !
	Hi.	Salut !
	Run!	Cours !
	Run!	Courez !
	Who?	Qui ?
	Wow!	Ça alors !


1. 下载数据集

```python
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```


2. 预处理数据集
步骤包括：我们用空格代替*不间断空格*（non-breaking space），使用小写字母替换大写字母，并在单词和标点符号之间插入空格。
```python
#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

go .	va !
hi .	salut !
run !	cours !
run !	courez !
who ?	qui ?
wow !	ça alors !

3. 词元化
之前都是字符词元化，现在是单词词元化。每个词元要不是个单词，要不是个符号。
此函数返回两个词元列表：`source`和`target`：
`source[i]`是源语言（这里是英语）第$i$个文本序列的词元列表，
`target[i]`是目标语言（这里是法语）第$i$个文本序列的词元列表。
```python
#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```


特殊词元：
将出现次数少于2次的低频率词元视为相同的未知词元（`<unk>`）,
在小批量时用于将序列填充到相同长度的填充词元（`<pad>`），
以及序列的开始词元（`<bos>`）和结束词元（`<eos>`）。

4.  处理文本序列（为了相同的长度）

为了相同的长度，我们要对序列样本有一个固定的长度。通过截断(truncation)和填充(paddinng)
- 截断：只取其前`num_steps` 个词元，并且丢弃剩余的词元。
- 填充：文本序列的词元数目少于`num_steps`时，我们将继续在其末尾添加特定的（`<pad>`）词元。

```python
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

5. 转换成小批量输入数据
将特定的（`<eos>`）词元添加到所有序列的末尾，用于表示序列的结束。
```python
#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
```

6. 定义迭代器

```python
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

7.  尝试读出这个小批量数据
```python
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
```
X: tensor([[ 7, 43,  4,  3,  1,  1,  1,  1],
        [44, 23,  4,  3,  1,  1,  1,  1]], dtype=torch.int32)
X的有效长度: tensor([4, 4])
Y: tensor([[ 6,  7, 40,  4,  3,  1,  1,  1],
        [ 0,  5,  3,  1,  1,  1,  1,  1]], dtype=torch.int32)
Y的有效长度: tensor([5, 3])

# 编码器-解码器架构
其输入和输出都是长度可变的序列。为了处理这种类型的输入和输出，于是设计出以下架构：
*编码器*（encoder）：变长序列（可变长度）输入转化成固长编码。
*解码器*（decoder）：固长编码状态转化成变长序列。

这被称为*编码器-解码器*（encoder-decoder）架构：
![[Pasted image 20231106153330.png]]
编码器-解码器架构的一个典型应用是机器翻译，其中输入是源语言的句子，编码器将其转换为上下文向量，解码器将这个上下文向量转换为目标语言的句子。

注意：为了逐个地生成长度可变的词元序列，解码器在每个时间步都会将输入（例如：在前一时间步生成的词元）和编码后的状态映射成当前时间步的输出词元。

```python
from torch import nn

# 定义编码器类
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    # forward方法用于对输入进行编码，具体的编码操作需要在子类中实现
    def forward(self, X, *args):
        raise NotImplementedError

# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    # init_state方法用于初始化解码器的状态，通常与编码器的输出相关
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    # forward方法用于对解码器的输入进行解码，具体的解码操作需要在子类中实现
    def forward(self, X, state):
        raise NotImplementedError

# 定义EncoderDecoder类，这是编码器-解码器架构的基类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder  # 初始化编码器
        self.decoder = decoder  # 初始化解码器

    # forward方法执行整个编码器-解码器过程，首先对输入进行编码，然后初始化解码器状态，最后对解码器的输入进行解码
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)  # 编码输入
        dec_state = self.decoder.init_state(enc_outputs, *args)  # 初始化解码器状态
        return self.decoder(dec_X, dec_state)  # 解码输入

```



# 序列到序列学习(seq2seq)


Seq2seq（Sequence-to-Sequence）是一种深度学习模型架构，用于处理序列到序列的任务。该模型最初由Google的研究员开发，主要用于将一个序列转换成另一个序列，通常涉及将输入序列转换为输出序列的不同长度和结构。**Seq2seq模型的核心思想是使用两个循环神经网络（RNN）：编码器（Encoder）和解码器（Decoder）**。

编码器负责将输入序列（如源语言文本）编码成一个固定长度的上下文向量（context vector），捕捉输入序列的语义信息。解码器则使用上下文向量来生成目标序列（如目标语言文本）。
![[Pasted image 20231106155808.png]]

几个特定设计：
1. 特定的`<eos>`表示序列结束词元。一旦**输出序列**生成此词元，模型就会停止预测。
2. 特定的`<bos>`表示序列开始词元，它是**解码器的输入序列**的第一个词元。
3. 使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。
4. 可以允许标签成为原始的输出序列。

## 编码器

编码器将长度可变的输入序列转换成形状固定的上下文变量$\mathbf{c}$，并且将输入序列的信息在该上下文变量中进行编码。
接下来我们使用单层循环神经网络来设计编码器:
考虑由一个序列组成的样本（批量大小是$1$）。假设输入序列是$x_1, \ldots, x_T$，其中$x_t$是输入文本序列中的第$t$个词元。在时间步$t$，循环神经网络将词元$x_t$的输入特征向量$\mathbf{x}_t$和$\mathbf{h} _{t-1}$（即上一时间步的隐状态）转换为$\mathbf{h}_t$（即当前步的隐状态）：

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

随后编码器通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量：

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$
当选择$q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$时，上下文变量仅仅是输入序列在最后时间步的隐状态$\mathbf{h}_T$。


```python
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状：(num_steps, batch_size, num_hiddens)
        # state的形状：(num_layers, batch_size, num_hiddens)
        return output, state
```


- 构造函数`__init__`接受以下参数：
  - `vocab_size`: 词汇表大小，用于嵌入层的输入维度。
  - `embed_size`: 嵌入维度的大小。
  - `num_hiddens`: 循环神经网络隐藏单元的数量。
  - `num_layers`: 循环神经网络的层数。
  - `dropout`: 可选参数，用于指定在循环神经网络中应用的丢弃率。

- 在构造函数中，初始化了两个关键组件：
  - `self.embedding`：嵌入层，用于将输入序列中的词汇索引映射成密集的词嵌入向量。
  - `self.rnn`：循环神经网络 (GRU)，用于对嵌入后的输入序列进行编码。

- `forward` 方法实现了编码器的前向传播过程：
  - 首先，输入`X`（形状：(batch_size, num_steps)）经过嵌入层，将词汇索引映射为嵌入向量，得到`X`的形状为(batch_size, num_steps, embed_size)。
  - 接着，对`X`进行维度转置，以适应循环神经网络的输入要求，得到的`X`的形状为(num_steps, batch_size, embed_size)。
  - 循环神经网络（GRU）处理输入序列`X`，得到输出`output`（形状：(num_steps, batch_size, num_hiddens)）和最终状态`state`（形状：(num_layers, batch_size, num_hiddens）。
  - 最后，`forward` 方法返回编码器的输出`output`和最终状态`state`。


随后实例化：


```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
```

- 创建了一个名为`encoder`的`Seq2SeqEncoder`实例，其中的参数如下：
  - `vocab_size=10`: 词汇表大小为10。
  - `embed_size=8`: 嵌入维度为8。
  - `num_hiddens=16`: 循环神经网络隐藏单元的数量为16。
  - `num_layers=2`: 循环神经网络的层数为2。

```python
encoder.eval()
```

- 将编码器的模式设置为评估模式。在评估模式下，模型不会进行梯度计算，通常用于推理或测试阶段。

```python
X = torch.zeros((4, 7), dtype=torch.long)
```

- 创建了一个名为`X`的张量，形状为(4, 7)，数据类型为`torch.long`，并且所有元素初始化为零。这个`X`表示一个批次大小为4，时间步数为7的输入序列。

```python
output, state = encoder(X)
```

- 将输入序列`X`传递给编码器`encoder`，并获取编码器的输出`output`和最终状态`state`。

由于这里使用的是门控循环单元，所以在最后一个时间步的多层隐状态的形状是（隐藏层的数量，批量大小，隐藏单元的数量）。如果使用长短期记忆网络，`state`中还将包含记忆单元信息。


## 解码器

对于每个时间步$t'$（与输入序列或编码器的时间步$t$不同），**解码器输出$y_{t'}$的概率**取决于先前的输出子序列$y_1, \ldots, y_{t'-1}$和上下文变量$\mathbf{c}$，即求$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$。

解码器：
在输出序列上的任意时间步$t^\prime$
循环神经网络输入是来自上一时间步的输出$y_{t^\prime-1}$和上下文变量$\mathbf{c}$。随后在当前时间步将它们和上一隐状态$\mathbf{s}_{t^\prime-1}$转换为隐状态$\mathbf{s}_{t^\prime}$。
因此，可以使用函数$g$来表示解码器的隐藏层的变换：

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$

获得解码器的隐状态之后，使用输出层和softmax操作，来计算在时间步$t^\prime$时输出$y_{t^\prime}$的条件概率分布
$P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$。

初始化：当实现解码器时，我们直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态。于是要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元。为了进一步包含经过编码的输入序列的信息，上下文变量在所有的时间步与解码器的输入进行拼接（concatenate）。为了预测输出词元的概率分布，在循环神经网络解码器的最后一层使用全连接层来变换隐状态。




```python
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状：(batch_size, num_steps, vocab_size)
        # state的形状：(num_layers, batch_size, num_hiddens)
        return output, state
```


- 构造函数`__init__`接受以下参数：
  - `vocab_size`: 词汇表大小，用于嵌入层的输出维度。
  - `embed_size`: 嵌入维度的大小。
  - `num_hiddens`: 循环神经网络隐藏单元的数量。
  - `num_layers`: 循环神经网络的层数。
  - `dropout`: 可选参数，用于指定在循环神经网络中应用的丢弃率。

- 在构造函数中，初始化了三个关键组件：
  - `self.embedding`：嵌入层，用于将输出序列 （target）中的词汇索引映射成密集的词嵌入向量。
  - `self.rnn`：循环神经网络 (GRU)，用于对嵌入后的输出序列进行解码。
  - `self.dense`：全连接层，将循环神经网络的输出映射为词汇表大小的向量，以便进行词汇的预测。

- `init_state` 方法用于初始化解码器的初始状态，通常使用编码器的输出状态来初始化。

- `forward` 方法实现了解码器的前向传播过程：
  - 首先，输入`X`（形状：(batch_size, num_steps)）经过嵌入层，将词汇索引映射为嵌入向量，然后通过维度转置，得到的`X`的形状为(num_steps, batch_size, embed_size)。
  - 然后，计算广播的上下文向量`context`，以便与输入`X`的形状相匹配。
  - 将嵌入后的输入`X`和上下文向量`context`连接在一起，得到`X_and_context`，然后将其输入到循环神经网络中。
  - 循环神经网络（GRU）处理`X_and_context`，得到输出`output`和最终状态`state`。
  - 最后，通过全连接层将`output`映射为词汇表大小的向量，得到`output`的形状为(batch_size, num_steps, vocab_size)，并返回输出`output`和最终状态`state`。

![[Pasted image 20231106163322.png]]


## 损失函数
在每个时间步，解码器预测了输出词元的概率分布。类似于语言模型，可以使用softmax来获得分布，并通过计算交叉熵损失函数来进行优化。**但是我们应该将填充词元的预测排除在损失函数的计算之外。**
例如，如果两个序列的有效长度（不包括填充词元）分别为$1$和$2$，则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为零。


```python
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)  # 获取输入序列的最大长度
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # 创建一个掩码矩阵，mask[i, j]为True表示第i个序列的第j个元素有效，否则为False

    X[~mask] = value  # 将不相关的项设置为指定的值
    return X

```

扩展softmax交叉熵损失函数来遮蔽不相关的预测。
方法：最初，所有预测词元的掩码都设置为1。一旦给定了有效长度，与填充词元对应的掩码将被设置为0。最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。

代码：
定义了一个名为`MaskedSoftmaxCELoss`的自定义损失函数类，用于计算带有遮蔽（masking）的 softmax 交叉熵损失。这种损失常用于序列到序列学习中，以处理序列长度不一的情况。下面是对代码的详细解释：

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的 softmax 交叉熵损失函数"""
    # pred的形状：(batch_size, num_steps, vocab_size)
    # label的形状：(batch_size, num_steps)
    # valid_len的形状：(batch_size,)

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)  # 创建一个与标签相同形状的权重张量，初始值为1
        weights = sequence_mask(weights, valid_len)  # 使用有效长度进行遮蔽

        self.reduction = 'none'  # 设置损失的计算方式为不进行降维
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 调用 nn.CrossEntropyLoss 的 forward 方法计算未加权的交叉熵损失，需要对 pred 进行维度转置

        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # 将未加权的损失乘以权重并计算平均值，以获得加权的损失

        return weighted_loss
```


- `forward` 方法接受三个参数：
  - `pred`：模型的预测，形状为 `(batch_size, num_steps, vocab_size)`，表示模型对每个时间步和每个词汇的预测概率分布。
  - `label`：标签，形状为 `(batch_size, num_steps)`，表示实际的标签序列。
  - `valid_len`：有效长度，形状为 `(batch_size,)`，表示每个序列的有效长度，通常由序列长度或者遮蔽信息生成。

1. 代码创建一个权重张量 `weights`，其形状与 `label` 相同，初始值都为 1。

2. 使用 `sequence_mask` 函数，根据 `valid_len` 来遮蔽 `weights`，将不相关的元素（即超过有效长度的部分）设置为0，以便在计算损失时不考虑这些部分。

3. 将损失计算方式 `self.reduction` 设置为 `'none'`，这表示不进行降维。

4. 使用 `super(MaskedSoftmaxCELoss, self).forward` 调用基类 `nn.CrossEntropyLoss` 的 `forward` 方法，计算未加权的交叉熵损失。在这之前，需要对 `pred` 进行维度转置，因为 `nn.CrossEntropyLoss` 期望输入维度为 `(batch_size, vocab_size, num_steps)`。

5.将未加权的损失乘以权重 `weights`，然后在 `dim=1`（时间步维度）上计算平均值，得到加权的损失，并返回该损失。

## 训练

解码器输入：特定的序列开始词元（`bos`）和原始的输出序列（不包括序列结束词元），这也指的强制教学。


```python
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    
    # 初始化权重
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)  # 使用 Xavier 初始化权重
    net.to(device)  # 将模型移动到指定的计算设备（CPU 或 GPU）
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用 Adam 优化器
    loss = MaskedSoftmaxCELoss()  # 使用自定义的带遮蔽的 softmax 交叉熵损失
    net.train()  # 设置模型为训练模式
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 用于累积训练损失总和和词元数量
        
        for batch in data_iter:
            optimizer.zero_grad()  # 梯度清零
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)  # 前向传播
            l = loss(Y_hat, Y, Y_valid_len)  # 计算损失
            l.sum().backward()  # 对损失函数的标量进行反向传播
            d2l.grad_clipping(net, 1)  # 梯度裁剪，防止梯度爆炸
            num_tokens = Y_valid_len.sum()
            optimizer.step()  # 更新模型参数
            
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

这个训练函数的主要功能包括：

- 初始化模型参数（权重）使用 Xavier 初始化方法，将模型移动到指定的计算设备（CPU 或 GPU）。
- 使用 Adam 优化器进行参数优化，使用自定义的带遮蔽的 softmax 交叉熵损失来计算损失。
- 将模型设置为训练模式，以便进行反向传播和梯度更新。
- 使用动画（`Animator`）来可视化损失的训练过程。
- 循环训练多个周期（epochs），每个周期中迭代处理数据集中的每个小批次（batch）：
  - 首先，将优化器的梯度清零，准备进行前向传播和反向传播。
  - 获取小批次数据，将其移动到指定的计算设备，创建解码输入（`dec_input`），并进行模型的前向传播。
  - 计算损失，将损失的标量进行反向传播。
  - 使用梯度裁剪，防止梯度爆炸。
  - 更新模型参数，通过优化器进行梯度下降。
  - 记录损失和词元数量。
- 在每个周期结束时，使用动画可视化损失的训练过程。

于是训练：
```python
# 导入必要的库和模块
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# 加载用于训练的数据集和构建源语言和目标语言的词汇表
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 创建编码器（Seq2SeqEncoder）和解码器（Seq2SeqDecoder）实例
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)

# 组合编码器和解码器成一个完整的序列到序列模型
net = d2l.EncoderDecoder(encoder, decoder)

# 调用训练函数 train_seq2seq 训练模型
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

```
![[Pasted image 20231106164511.png]]

## 预测
![[Pasted image 20231106164833.png]]
```python
# 导入必要的库和模块

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    
    # 在预测时将网络设置为评估模式
    net.eval()
    
    # 将源语句分割成词元并添加结束词元
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    
    # 计算源语句的有效长度
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    
    # 截断或填充源语句，以满足指定的时间步数
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    
    # 添加批量维度，将源语句转换为张量
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    # 使用编码器对源语句进行编码，获取编码器的输出
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    
    # 初始化解码器的状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    # 创建解码器的输入张量，初始词元为'<bos>'（开始词元）
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    
    # 开始生成目标语句的词元序列
    for _ in range(num_steps):
        # 使用解码器进行预测，获取输出和新的解码器状态
        Y, dec_state = net.decoder(dec_X, dec_state)
        
        # 选择具有最高可能性的词元作为下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
        # 如果需要保存注意力权重（稍后讨论），则将其保存到列表中
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        
        # 如果预测到了结束词元'<eos>'，则停止生成
        if pred == tgt_vocab['<eos>']:
            break
        
        # 将预测的词元添加到输出序列中
        output_seq.append(pred)
    
    # 将输出序列中的词元转换为字符串形式
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

```

1. 准备输入数据和模型
   - 将模型 `net` 设置为评估模式，以便进行预测。
   - 将源语句分割成词元，并将结束词元添加到词元序列中。
   - 计算源语句的有效长度，并将源语句截断或填充，以满足指定的时间步数。
   - 创建包含批量维度的输入张量 `enc_X`，将源语句转换为张量。

2. 生成目标语句
   - 使用编码器对源语句进行编码，获取编码器的输出。
   - 初始化解码器的状态，并创建解码器的输入张量，初始词元为 `<bos>`（开始词元）。
   - 开始生成目标语句的词元序列，通过循环迭代生成每个时间步的词元。
   - 使用解码器进行预测，获取输出和新的解码器状态。
   - 选择具有最高可能性的词元作为下一个时间步的输入。
   - 如果需要保存注意力权重（用于注意力机制的可视化），则将其保存到列表中。
   - 如果预测到了结束词元 `<eos>`，则停止生成，否则继续生成。
   - 将预测的词元添加到输出序列中。

3. 返回预测结果
   - 最后，将输出序列中的词元转换为字符串形式。
   - 返回生成的目标语句以及可选的注意力权重（如果需要）。

## 预测序列的评估

原则上说，对于预测序列中的任意$n$元语法（n-grams），
BLEU的评估都是这个$n$元语法是否出现在标签序列中。

BLEU的计算方法如下：

1. 对于每个句子或文本段的机器翻译，首先将其分成n-gram（通常是1-gram到4-gram，表示从单个单词到包含4个连续单词的片段）。

2. 然后，计算机器翻译中每个n-gram在参考翻译中出现的频率，以及机器翻译中的n-gram出现的频率。

3. 计算每个n-gram的精确匹配得分（Precision），它表示机器翻译中的n-gram在参考翻译中出现的频率与机器翻译中的n-gram出现的频率的比率。

4. BLEU计算的最终得分是通过计算机器翻译中每个n-gram的精确匹配得分的几何平均，并且还考虑了一个叫做“截断”的惩罚因子，用于惩罚较长的机器翻译文本。

BLEU的得分范围通常在0到1之间，越接近1表示机器翻译结果与参考翻译越相似。
我们将BLEU定义为：

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$


其中$\mathrm{len}_{\text{label}}$表示标签序列中的词元数和
$\mathrm{len}_{\text{pred}}$表示预测序列中的词元数，
$k$是用于匹配的最长的$n$元语法。
另外，用$p_n$表示$n$元语法的精确度，它是两个数量的比值：
第一个是预测序列与标签序列中匹配的$n$元语法的数量，
第二个是预测序列中$n$元语法的数量的比率。
具体地说，给定标签序列$A$、$B$、$C$、$D$、$E$、$F$
和预测序列$A$、$B$、$B$、$C$、$D$，
我们有$p_1 = 4/5$、$p_2 = 3/4$、$p_3 = 1/3$和$p_4 = 0$。



```python
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

这个函数采用三个参数：`pred_seq`（机器翻译的结果）、`label_seq`（参考翻译的结果）和 `k`（用于计算BLEU时的n-gram最大长度）。以下是步骤：

1. 将`pred_seq`和`label_seq`分割成单词（或标记）序列，通过空格分割。

2. 计算一个长度惩罚分数 `score`，该分数会惩罚机器翻译结果的长度与参考翻译长度之间的差异。这个惩罚分数通过 `math.exp` 函数计算，当机器翻译结果的长度小于或等于参考翻译长度时，得分为1，当机器翻译结果更短时，得分小于1。

3. 对于每个n-gram（从1-gram到`k`-gram），计算精确匹配的数量。首先，为参考翻译中的n-gram建立一个字典 `label_subs` 来统计它们的出现次数。然后，遍历机器翻译结果中的n-gram，如果在 `label_subs` 中存在相应的n-gram，则增加精确匹配的数量。

4. 使用一个几何平均来计算最终的BLEU分数，对每个n-gram的匹配得分进行加权，且每个n-gram的权重以指数衰减。这个几何平均的结果就是最终的BLEU分数。

最终的BLEU分数会在0到1之间，越接近1表示机器翻译结果与参考翻译越相似。

## 预测评分应用

```python
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

go . => va !, bleu 1.000
i lost . => j'ai perdu ., bleu 1.000
he's calm . => il est riche ., bleu 0.658
i'm home . => je suis en retard ?, bleu 0.447

# 束搜索
https://zhuanlan.zhihu.com/p/82829880
贪心搜索、穷举搜索，束搜索。

## 贪心搜索

对于输出序列的每一时间步$t'$，
我们都将基于贪心搜索从$\mathcal{Y}$中找到具有最高条件概率的词元，即：

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$
一旦输出序列包含了`<eos>`或者达到其最大长度$T'$，则输出完成。在每个时间步，贪心搜索选择具有最高条件概率的词元。

然而*最优序列*（optimal sequence）应该是最大化$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$值的输出序列，这是基于输入序列生成输出序列的条件概率。然而，贪心搜索无法保证得到最优序列。

##  穷举搜索

如果目标是获得最优序列，我们可以考虑使用*穷举搜索*（exhaustive search）：
穷举地列举所有可能的输出序列及其条件概率，然后计算输出条件概率最高的一个。但是计算量超级大。

## 束搜索
如果精度最重要，则显然是穷举搜索。如果计算成本最重要，则显然是贪心搜索。而束搜索的实际应用则介于这两个极端之间。

束搜索它有一个超参数，名为*束宽*（beam size）$k$。
在时间步$1$，我们选择具有最高条件概率的$k$个词元。这$k$个词元将分别是$k$个候选输出序列的第一个词元。在随后的每个时间步，基于上一时间步的$k$个候选输出序列，我们将继续从$k\left|\mathcal{Y}\right|$个可能的选择中挑出具有最高条件概率的$k$个候选输出序列。

![[Pasted image 20231106170747.png]]


