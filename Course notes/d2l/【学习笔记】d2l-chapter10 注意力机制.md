# 前言
1. 经典注意力框架：
   - 回顾了一个经典的注意力框架，最主要的是里面的注意力提示（attention cues）。
   - 解释了如何在视觉场景中展开注意力。这通常涉及根据特定的注意力提示来聚焦于感兴趣的区域或特征。

2. Nadaraya-Watson核回归（kernel regression）：
   - 提到这是一个机器学习的简单演示，具有注意力机制。
   - Nadaraya-Watson核回归是一种用于回归分析的方法，通常用于估计条件期望函数。

4. 注意力函数和Bahdanau注意力：
   - 提到了注意力函数在深度学习中的广泛应用。
   - 具体介绍了如何使用这些函数来设计Bahdanau注意力。
   - Bahdanau注意力是一种具有突破性价值的注意力模型，它具备双向对齐和微分性质，常用于序列到序列模型中，如机器翻译。

5. Transformer架构：
   - 描述了基于注意力机制的Transformer架构。
   - 提到了多头注意力和自注意力的使用。
   - Transformer是一种现代深度学习架构，自2017年问世以来在多个领域中广泛应用，包括自然语言处理、计算机视觉、语音识别和强化学习。

# 注意力提示 attention cues
注意力在视觉世界中：双组件框架。在这个框架中，受试者基于*非自主性提示*和*自主性提示*
有选择地引导注意力的焦点。 非自主性提示，比如红色的比较吸引人眼球。自主性提示，比如自己靠自我意识主动的focus在一本书上。

基于attention cues，于是我们开发出基于神经网络的注意力机制框架。
![[Pasted image 20231106141921.png]]
图上包含了一下信息：
- 查询query（自主提示）和键key（非自主提示）之间的交互形成了注意力汇聚。
- 注意力汇聚有选择地聚合了值（感官输入）以生成最终的输出。
- 键值是成对的。


# 注意力汇聚——Nadaraya-Waston 核回归

回归问题：
给定的成对的“输入－输出”数据集$\{(x_1, y_1), \ldots, (x_n, y_n$}$，如何学习$f$来预测任意新输入$x$的输出$\hat{y} = f(x)$？

这里的问题，我们只有数据，不知道模型。不知道它是线性，还是非线性，不知道它的具体模式。
***
生成数据集：
根据下面的非线性函数生成一个人工数据集，其中加入的噪声项为$\epsilon$：

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

其中$\epsilon$服从均值为$0$和标准差为$0.5$的正态分布。生成了$50$个训练样本和$50$个测试样本。
为了更好地可视化之后的注意力模式，需要将训练样本进行排序。
***
解决上述的回归问题，共有？个方案。分别是：1.平均汇聚 2.非参数注意力汇聚 3.带参数注意力汇聚

## 平均汇聚
首先是最简单的估计器：平均汇聚。
基于平均汇聚来计算所有训练样本输出值的平均值：

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
随后这个$f(x)$ 就是预测的数值。
![[Pasted image 20231107145844.png]]


## 非参数注意力汇聚
 上文的平均汇聚可以看到和输入x根本没有关系，无论输入是多少，输出永远是一个值。这样就忽略了一个有效信息。于是Nadaraya-Watson核回归被提出(1964)。
 Nadaraya-Watson核回归的主要思想是通过使用核函数（kernel function）来对每个数据点进行加权，然后将这些加权的数据点的输出值进行加权平均，以获得对输出变量的估计。
 
 通俗来说：在平均汇聚的基础上，根据输入的位置对输出$y_i$进行加权：

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
	其中$K$是*核*（kernel）。
受此启发，可以把上述公式转化成更一般的形式，简称*注意力汇聚*（attention pooling）公式：

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$

其中$x$是查询，$(x_i, y_i)$是键值对。注意力汇聚是$y_i$的加权平均。
将查询$x$和键$x_i$之间的关系建模为*注意力权重*（attention weight）$\alpha(x, x_i)$，这个权重将被分配给每一个对应值$y_i$。
***
我们代入一个*高斯核*（Gaussian kernel），其定义为：

$$K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).$$

得到：
$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$

如果一个键$x_i$越是接近给定的查询$x$，那么分配给这个键对应值$y_i$的注意力权重就会越大，也就“获得了更多的注意力”。

![[Pasted image 20231109102021.png]]

## 带参数注意力汇聚
非参数的Nadaraya-Watson核回归以其**一致性**而著称，即在数据充足时会收敛到最优结果。但是可以将可学习的参数集成到注意力汇聚中来进一步改进该模型。

我们引入一个可学习参数$w$，用以调整查询$x$和键$x_i$之间的距离：

$$
\begin{aligned}
f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\
&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\
&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.
\end{aligned}
$$


接下来，我们将通过训练来学习注意力汇聚的参数。

### 批量矩阵乘法

为了高效计算小批量数据的注意力，我们可以利用深度学习框架中提供的批量矩阵乘法。假设第一个小批量数据包含$n$个矩阵$\mathbf{X}_1,\ldots, \mathbf{X}_n$，形状为$a\times b$，第二个小批量包含$n$个矩阵$\mathbf{Y}_1, \ldots, \mathbf{Y}_n$，形状为$b\times c$。它们的批量矩阵乘法得到$n$个矩阵$\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$，形状为$a\times c$。因此，我们假定两个张量的形状分别是$(n,a,b)$和$(n,b,c)$，它们的批量矩阵乘法输出的形状为$(n,a,c)$。

```python
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```python
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```
![[Pasted image 20231109104623.png]]


# 注意力评分函数


之前，我们使用高斯核来建模查询和键之间的关系。式子中高斯核**指数部分**($-\frac{u^2}{2}$)被称为*注意力评分函数*，简称*评分函数*。这个函数的输出被输入到 softmax 函数中，得到注意力权重(即概率分布)，最终通过这些权重对值进行加权求和得到注意力汇聚的输出。
![[Pasted image 20231109105359.png]]


注意力评分函数是注意力机制（attention mechanism）中的一个关键组成部分。注意力机制允许模型在处理序列数据时集中关注输入的不同部分，而不是简单地平均考虑所有输入。

在注意力机制中，有一个注意力评分函数，它计算一个分数，表示模型在处理当前位置时对输入序列中不同位置的关注程度。这个分数通常通过某种函数计算，函数的选择取决于具体的注意力机制设计。


用数学语言：假设有一个查询 $\mathbf{q} \in \mathbb{R}^q$ 和 $m$ 个“键－值”对 $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$，其中 $\mathbf{k}_i \in \mathbb{R}^k$，$\mathbf{v}_i \in \mathbb{R}^v$。
注意力汇聚函数 $f$ 被表示为值的加权和：

$$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$$

其中查询 $\mathbf{q}$ 和键 $\mathbf{k}_i$ 的注意力权重（标量）通过注意力评分函数 $a$ 映射两个向量为标量，然后经过 softmax 运算得到：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.$$

*常见的注意力评分函数包括*：

1. **点积注意力（Dot Product Attention）**:
   $$ \text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q} \cdot \mathbf{k} $$
2. **缩放点积注意力（Scaled Dot Product Attention）**:
   $$ \text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}} $$
   其中，$d$ 是注意力向量的维度。
3. **加性注意力（Additive Attention）**:
   $$ \text{score}(\mathbf{q}, \mathbf{k}) = \text{tanh}(\mathbf{W}_1 \mathbf{q} + \mathbf{W}_2 \mathbf{k}) $$
   其中，$\mathbf{W}_1$和 $\mathbf{W}_2$ 是可学习的权重矩阵。

4. **多头注意力（Multi-head Attention）**:
   在这种情况下，模型使用多个注意力头，每个头都有自己的注意力评分函数，最后将它们的输出合并起来。
## 掩蔽softmax操作

如前所述，softmax 操作用于生成一个注意力权重的概率分布。但有些值（比如序列中包含填充项）不应该参加到注意力汇聚中。

*基本思想*：是在进行softmax计算之前，指定一个有效序列长度（即词元的个数），在计算 softmax 时过滤掉超出指定范围的位置，随后将填充项的位置的权重设置为一个**极大的负数**，使得经过softmax后的概率趋近于零。这样，填充项对模型输出的影响就会被最小化。

*示例*：其中使用掩蔽softmax来处理序列中的填充项。假设我们有一个序列 $[x_1, x_2, x_3, \text{<padding>}, \text{<padding>}]$，其中 $<padding>$表示填充项。softmax前的未掩蔽分数为 $[a, b, c, d, e]$，掩蔽后的分数为 $[a, b, c, -\infty, -\infty]$。然后，通过softmax操作，模型会将注意力更集中在实际序列元素上，而不是填充项上。



下面的 `masked_softmax` 函数实现了这种*掩蔽 softmax 操作*，其中任何超出有效长度的位置都被掩蔽并置为0。

```python
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X: 3D张量，valid_lens: 1D或2D张量
    
    # 如果没有提供有效长度信息，直接使用PyTorch的softmax函数
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        
        # 如果有效长度是1D张量，将其广播到匹配X的形状
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

```



## 加性注意力（additive attention）

一般来说，当查询和键是不同长度的矢量时，可以使用加性注意力作为评分函数。
*定义*：通过对查询向量（query vector）和键向量（key vector）进行加法运算得。
*数学公式*：给定查询$\mathbf{q} \in \mathbb{R}^q$和键$\mathbf{k} \in \mathbb{R}^k$，评分函数为
$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$

可学习的参数是$\mathbf W_q\in\mathbb R^{h\times q}$、$\mathbf W_k\in\mathbb R^{h\times k}$和$\mathbf w_v\in\mathbb R^{h}$。

*具体实现*：将查询和键连结起来后输入到一个多层感知机（MLP）中，感知机包含一个隐藏层，其隐藏单元数是一个超参数$h$。通过使用$\tanh$作为激活函数，并且禁用偏置项。
```python
#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 使用线性层定义注意力模型的参数
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # 定义dropout层，以防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 将queries和keys分别传入线性层进行变换
        queries, keys = self.W_q(queries), self.W_k(keys)
        
        # 在维度上进行扩展，以便后续广播求和
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # keys的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        #(batch_size, 查询的个数, 1, 1, num_hidden)+(batch_size, 1, 1, “键－值”对的个数, num_hidden)
        features = torch.tanh(features)
        
        # 使用线性层计算注意力分数，去除最后一维
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        
        # 使用掩蔽softmax计算注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # 在“键－值”对维度上进行加权求和，得到最终输出
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

```


## 缩放点积注意力


在缩放点积注意力中，首先计算查询（query）、键（key）和值（value）的三个矩阵，然后通过计算查询和键的点积，再进行缩放操作，最后将结果与值的矩阵相乘，得到最终的注意力权重。这个过程可以用以下公式表示：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

其中，$Q$是查询矩阵，$K$ 是键矩阵，$V$ 是值矩阵，$d_k$ 是键的维度。缩放操作通过除以 $\sqrt{d_k}$ 来平衡点积的数量级，有助于确保梯度在反向传播时不会变得太大。

>为什么要除以$\sqrt{d}$？

因为式子基于以下假设：假设查询和键的所有元素都是独立的随机变量，并且都满足零均值和单位方差，那么两个向量的点积的均值为$0$，方差为$d$。为确保无论向量长度如何，点积的方差在不考虑向量长度的情况下仍然是$1$，我们再将点积除以$\sqrt{d}$.

```python
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

# Bahdanau 注意力

以前：在传统Seq2Seq模型中，编码器（Encoder）将**输入序列编码**为一个**固定长度**的上下文向量，然后解码器（Decoder）使用这个上下文向量生成输出序列。


- *创新点*：在传统的注意力机制中，模型会对输入序列的所有位置计算权重，然后将这些权重用于加权求和。而在Bahdanau 注意力中，**注意力权重的计算是动态的，使得解码器在生成每个输出的时候关注输入序列的不同部分**。具体而言，**Bahdanau 注意力使用了一个可学习的参数向量，被称为上下文向量（context vector），用于计算每个输入位置的注意力权重**。
***
>通俗的解释上下文向量(context vector)
>当我们在处理序列数据时，比如翻译一句话，模型需要关注输入序列的不同部分以正确生成输出。上下文向量就像是一个动态的摘要，根据当前生成的部分输出和输入序列的不同位置信息，帮助模型决定在每个时刻应该关注输入序列的哪些部分。你可以把上下文向量想象成一个“注意力背后的助手”，在每个步骤都告诉模型：“嘿，在这个时刻，你应该更关注输入的这一部分，因为它与当前输出的内容更相关。”
***

Bahdanau 注意力模型与[[【学习笔记】d2l-chapter9 现代循环神经网络#序列到序列学习(seq2seq)]]的架构一模一样。改的就是解码器。

## 计算公式

1. **计算未缩放的注意力分数（Attention Score）：**

   首先，对于给定的解码器的当前时间步 $t$和编码器的每个时间步 $i$，计算未缩放的注意力分数：
   $$e_{t,i} = \text{score}(s_{t-1}, h_i)$$
   其中，$s_{t-1}$是解码器上一个时间步的隐藏状态，$h_i$ 是编码器在时间步 $i$的隐藏状态，$\text{score}$是一个用于计算注意力分数的函数，可以是点积、缩放点积等。
2. **应用softmax函数得到注意力权重：**

   使用softmax函数将未缩放的注意力分数转换为注意力权重。这样可以确保注意力权重的总和为1，表示对输入序列的加权和。
   $$a_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}$$
   其中，$T$ 是输入序列的长度。
3. **计算上下文向量（Context Vector）：**

   最后，使用注意力权重对编码器的隐藏状态进行加权求和，得到解码器当前时间步 $t$的上下文向量：
   $$c_t = \sum_{i=1}^{T} a_{t,i} \cdot h_i$$

***
总结：
$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

时间步$t' - 1$时的解码器隐状态$\mathbf{s}_{t' - 1}$是查询。编码器隐状态$\mathbf{h}_t$既是键，也是值。注意力权重$\alpha$是加性注意力打分函数计算。
我们可以对比[[【学习笔记】d2l-chapter9 现代循环神经网络#编码器]]中的上下文向量c，不过这里的$c_t'$ 会替代c，在任意时间步的时候。
***
>编码器隐状态$\mathbf{h}_t$既是键，也是值?

在Bahdanau注意力中，编码器的隐藏状态（通常用 $\mathbf{h}_t$ 表示）既被用作计算注意力分数的"键"，也被用作加权求和的"值"。

1. **键（Key）的作用：**
   - 编码器隐藏状态作为"键"，用于度量解码器当前时间步的隐藏状态与编码器各个时间步的关联程度。

2. **值（Value）的作用：**
   - 编码器隐藏状态作为"值"，在计算注意力权重之后，用于加权求和。这样，得到的加权和就是解码器当前时间步的上下文向量。
总之，因为本来的c也就是由隐状态$h$组成的。现在只不过对于不同时刻，结合解码器隐状态，我对h的权重不同，所以1.求权重则作为key。2.本身的含义就是值。
***

![[Pasted image 20231109121344.png]]

## 代码
重新定义解码器（因为与9.7相比，增加了注意力机制）
```python
#@save
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

解码器初始化：
1. 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；
2. 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
3. 编码器有效长度（排除在注意力池中填充词元）。

在每个解码时间步骤中，解码器上一个时间步的最终层隐状态将用作查询（参考公式$s_{t-1}'$）。
因此，注意力输出和输入嵌入都连结为循环神经网络解码器的输入。

```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```


实例化训练：
```python
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

预测（英语翻译成法语）
```python
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```
# Luong 注意力
Luong 2015注意力是另一种用于序列到序列模型的注意力机制。类似于Bahdanau 注意力，Luong 注意力旨在解决传统Seq2Seq模型中固定长度上下文向量的问题，并提高处理长序列的能力。

与Bahdanau 注意力不同，Luong 提出了两种不同的注意力机制：Global Attention和Local Attention。

1. **Global Attention:**

   - **Score计算：** 在Global Attention中，注意力分数是通过计算当前解码器的隐藏状态与所有编码器的隐藏状态之间的关系来计算的。具体来说，给定解码器的当前隐藏状态 $s_t$ 和编码器的所有隐藏状态 $h_1, h_2, ..., h_T$，计算注意力分数的方式可以是点积、缩放点积等。

   - **权重计算：** 使用 softmax 函数将注意力分数转换为注意力权重。

   - **上下文向量：** 使用注意力权重对编码器的隐藏状态进行加权求和，得到上下文向量 $c_t$。

2. **Local Attention:**

   - 在Local Attention中，模型在当前时间步附近的**一个窗口内进行关注**。
  
   - 计算注意力分数时，引入了一个窗口函数（window function）来决定在哪个位置集中注意力。这样可以减少计算的复杂度，并且在处理长序列时仍然能够获得较好的性能。

总体来说，Luong 注意力与Bahdanau 注意力有一些相似之处，但在具体计算上有一些细微的差异，主要体现在注意力分数的计算和权重的计算上。

具体对比：
https://zhuanlan.zhihu.com/p/129316415
# 多头注意力
多头自注意力是一种用于深度学习中的注意力机制，它可以帮助模型在处理序列数据时更好地捕捉不同位置之间的关系。**它的主要思想是将输入数据分成多个头，每个头都可以学习到不同的特征，然后将这些头的输出合并起来，以获得更好的表示。**

以下是多头注意力的基本步骤：

1. **多头投影：** 输入序列通过h个独立的*线性投影*（linear projections）来变换查询、键和值，生成多个表示子空间，每个头有自己的权重矩阵。

2. **独立计算注意力：** 在每个表示子空间中，独立地计算注意力权重。这类似于单头注意力机制的计算，但是应用于多个表示子空间。（体现在图上的蓝色“注意力”）

3. **多头拼接：** 将h个头的注意力权重和值进行拼接。（体现在图上的连结）

4. **线性变换：** 将拼接后的结果通过另一个*线性变换映射*为最终的多头注意力输出。（体现在图上的全连接层）


对于$h$个注意力汇聚输出，每一个注意力汇聚都被称作一个*头*（head）。

![[Pasted image 20231109135121.png]]

数学语言：

给定查询$\mathbf{q} \in \mathbb{R}^{d_q}$、键$\mathbf{k} \in \mathbb{R}^{d_k}$和值$\mathbf{v} \in \mathbb{R}^{d_v}$，
每个注意力头$\mathbf{h}_i$（$i = 1, \ldots, h$）的计算方法为：
$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

其中，可学习的参数包括$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$，以及代表注意力汇聚的函数$f$(比如加性注意力和缩放点积注意力)。

多头注意力的输出需要经过另一个线性转换，它对应着$h$个头连结后的结果，因此其可学习参数是
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$：

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

基于这种设计，每个头都可能会关注输入的不同部分，
可以表示比简单加权平均值更复杂的函数。
## 实现
```python
# 定义一个多头注意力模型类，继承自nn.Module

class MultiHeadAttention(nn.Module):
    """多头注意力"""

    # 初始化方法，定义了参数和超参数
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads  # 头的数量
        self.attention = d2l.DotProductAttention(dropout)  # 关注机制对象
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  # 用于转换query的线性层
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)  # 用于转换key的线性层
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  # 用于转换value的线性层
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # 输出线性层

    # 前向传播方法，用于计算模型的输出
    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

这是一个多头注意力模型，它首先通过三个线性层将输入的queries、keys和values转换为适当大小的张量，然后使用一个关注机制来计算输出，最后再通过一个线性层得到最终的输出。在前向传播过程中，如果valid_lens不为空，则会将其按照头的数量进行重复操作。

对于上文的转置函数来说：
```python
#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

编写实例测试一下：
```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```
MultiHeadAttention(
  (attention): DotProductAttention(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (W_q): Linear(in_features=100, out_features=100, bias=False)
  (W_k): Linear(in_features=100, out_features=100, bias=False)
  (W_v): Linear(in_features=100, out_features=100, bias=False)
  (W_o): Linear(in_features=100, out_features=100, bias=False)
)

# 自注意力

https://zhuanlan.zhihu.com/p/619154409
一篇不错的关于自注意力的动态图理解。

*自注意力*（self-attention）：查询、键和值来自同一组输入。每个查询都会关注所有的键－值对并生成一个注意力输出。
*创新点*：通过将每个输入位置与其它位置进行比较，来计算它们之间的相关性。这种方法能够更高效地计算**长序列**之间的相互关系，并且在处理长句子时能够更好地捕捉语义信息。

## 数学公式
形式上，给定一个输入序列 $(x_1, x_2, ..., x_n)$，自注意力的计算过程如下：
1. **计算注意力分数（Attention Scores）：**
   对于序列中的每个位置 $i$，计算注意力分数 $e_{ij}$，表示位置 $i$ 与位置 $j$ 之间的关联程度。
   
   $$e_{ij} = \text{score}(x_i, x_j)$$

   这里的 $\text{score}$ 是用于计算注意力分数的函数，可以是点积、缩放点积等。

2. **计算注意力权重（Attention Weights）：**
   使用 softmax 函数将注意力分数转换为注意力权重，确保它们的总和为1。
   
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}$$

3. **计算加权和（Weighted Sum）：**
   使用注意力权重对序列中的所有位置进行加权求和，得到每个位置的输出。
   
  $$\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} x_j$$

对于每个时间步，直接计算它和其他序列位置的注意力分数（比如理解成相似度），然后相似度高的自然对应的值权重比例高。 所以它有利于保留长序列的语义信息。


总结成一个公式：
给定一个由词元组成的输入序列$\mathbf{x}_1, \ldots, \mathbf{x}_n$，其中任意$\mathbf{x}_i \in \mathbb{R}^d$（$1 \leq i \leq n$）。该序列的自注意力输出为一个长度相同的序列$\mathbf{y}_1, \ldots, \mathbf{y}_n$，其中：
$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$
f为注意力汇聚函数$f$。

 ## 比较CNN、RNN、自注意力

目标：将$n$ 个词元组成的序列映射到另一个长度相同的序列。其中输入和输出词元由$d$ 维向量表示。

1. **适用性：**
   - **CNN：** 主要用于处理具有局部结构的数据，如图像。在序列数据上的应用通常需要将序列转换为图像或矩阵形式，然后使用卷积进行特征提取。
   - **RNN：** 专门设计用于处理序列数据，能够捕捉时序信息。但在处理长序列时，可能会面临梯度消失或梯度爆炸的问题。
   - **自注意力：** 适用于各种序列数据，尤其在处理长距离依赖关系时效果显著。

2. **并行性：**
   - **CNN：** 具有很好的并行性，可以同时处理输入中的多个区域。
   - **RNN：** 在序列中的每个时间步都依赖于前一个时间步的计算，限制了并行性。
   - **自注意力：** 具有很好的并行性，可以同时关注序列中的所有位置，因此在计算上更高效。

3. **长距离依赖关系：**
   - **CNN：** 在卷积核的感受野内可以捕捉到局部关系，但对于全局的长距离依赖关系有限。
   - **RNN：** 由于梯度传播的原因，处理长距离依赖关系时可能面临困难。
   - **自注意力：** 可以轻松捕捉长距离依赖关系，每个位置都可以关注序列中其他位置的信息。

4. **参数共享：**
   - **CNN：** 利用卷积核实现参数共享，减少模型参数量。
   - **RNN：** 参数共享较少，需要在不同的时间步上分别计算权重。
   - **自注意力：** 具有全连接性，每个位置都可以与所有其他位置发生联系，参数共享较少。

5. **处理速度：**
   - **CNN：** 由于并行性较强，处理速度通常较快。
   - **RNN：** 由于时间步之间的依赖性，处理速度可能较慢。
   - **自注意力：** 具有很好的并行性，可以提高处理速度。

6. **灵活性：**
   - **CNN：** 对于**固定大小**的输入较为灵活，但对序列长度不敏感。
   - **RNN：** 对于**可变长度**的序列较为灵活，但在处理长序列时可能受限。
   - **自注意力：** 在处理序列时非常灵活，适用于各种序列任务和长度。


随后我们用数据去衡量，比较**计算复杂度，顺序操作、最大路径长度**。
CNN：假设卷积核大小为$k$的卷积层。（处理序列可能是一维卷积？）由于序列长度是$n$，输入通道和输出通道数都是$d$，则计算复杂度为$O(knd^2)$、循序操作为$O(1)$、最大路径长度为$O(n/k)$。

RNN：计算复杂度为$O(nd^2)$、循序操作为$O(n)$且无法并行化、最大路径长度为$O(n)$。

自注意力：计算复杂度为$O(n^2d)$、循序操作为$O(1)$、最大路径长度为$O(n)$。



![[Pasted image 20231109185250.png]]


## 位置编码
使用位置编码的原因：自注意力则因为并行计算而放弃了顺序操作。所以对于它所计算的位置都要有一个编码。

为了使用序列的顺序信息，通过在输入表示中添加*位置编码*（positional encoding）来注入绝对的或相对的位置信息。位置编码可以通过**学习得到**也可以**直接固定得到**。

***
[^1] https://zhuanlan.zhihu.com/p/352233973 知乎-详解自注意力机制中的位置编码(1)
[^2] https://zhuanlan.zhihu.com/p/354963727 知乎-详解自注意力机制中的位置编码(2)
以及它提到的相关参考文献。
里面解决了，为什么要用位置编码，为什么会用sin，cos。
***
>为什么位置编码需要学习得到？

因为位置你难以表达，比如图结构的位置，要如何表达？不如给它足够多的维度，让它自学习位置，这个学到的位置或许是它结构序列的一个量化体现。

***


假设输入表示$\mathbf{X} \in \mathbb{R}^{n \times d}$包含一个序列中$n$个词元的$d$维嵌入表示。位置编码使用相同形状的位置嵌入矩阵$\mathbf{P} \in \mathbb{R}^{n \times d}$输出$\mathbf{X} + \mathbf{P}$，矩阵第$i$行、第$2j$列和$2j+1$列上的元素为[^1]：

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
这个公式的核心思想是使用正弦和余弦函数在不同频率上对位置进行编码。通过这种方式，不同位置的编码将在隐藏层的维度上有所不同，为模型提供了序列中单词或标记的位置信息。


虽然这个三角函数位置编码很奇怪，但我们可以在PositioinEncoding中先实现它：
```python
#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

### 绝对位置信息
1. 直觉是为位置信息顺序编码，比如[1,2,3,4,5......],但是这会带来1.最后的非常大，比如1千万。2.训练的时候长度只有10，但预测的时候却出现了位置100。所以不行。
2. 采用二进制。
```python
for i in range(8):
    print(f'{i}的二进制是：{i:>03b}')
```
0的二进制是：000
1的二进制是：001
2的二进制是：010
3的二进制是：011
4的二进制是：100
5的二进制是：101
6的二进制是：110
7的二进制是：111

可以看到较高位的交替频率低于较低位。然而，这是整数，我们现在有浮点数！我们可以更加的节约空间。float continous counterparts 对应于Sinusoidal functions。
此外，通过降低频率，可以从红色位变为橙色位。
![[Pasted image 20231111203253.png]]
### 相对位置信息
任何确定的位置偏移$\delta$，位置$i + \delta$处的位置编码可以线性投影位置$i$处的位置编码来表示。

令$\omega_j = 1/10000^{2j/d}$，对于任何确定的位置偏移$\delta$，任何一对$(p_{i, 2j}, p_{i, 2j+1})$都可以线性投影到
$(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$：

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

$2\times 2$投影矩阵不依赖于任何位置的索引$i$。

具体的严谨推导请见
https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/

# Transformer
https://zhuanlan.zhihu.com/p/338817680 这个讲的也不错。

由于自注意力同时具有并行计算和最短的最大路径长度这两个优势。所以想到用Transformer来做深度学习架构。Transformer模型完全基于注意力机制，没有任何卷积层或循环神经网络层。

![[Pasted image 20231109190737.png]]

1. 数据在输入编码器和解码器之前，都要经过：
	词嵌入层
	位置编码层
	编码器堆栈包含若干个编码器。
2. 每个编码器都包含：
	多头注意层
	前馈层
	解码器堆栈包含若干个解码器。
3. 每个解码器都包含：
	两个多头注意层
	前馈层
4. 输出产生最终输出：
	线性层
	Softmax层

1. 嵌入层（Embedding）
Transformer 的编码器和解码器各有一个嵌入层（Embedding ）。输入序列被送入编码器的嵌入层，被称为输入嵌入（ Input Embedding）。
目标序列在右移一个位置，并在第一个位置插入一个Start token 后被送入解码器的嵌入层。注意，在推理过程中，我们没有目标序列，我们在一个循环中把输出序列送入解码器的嵌入层，正如第一部分中所提到的。这就是为什么它被称为 "输出嵌入"（ Output Embedding）。

文本序列被映射成词汇表的单词ID的数字序列。嵌入层再将每个数字序列射成一个嵌入向量，这是该词含义的一个更丰富的表示。

2. 位置编码（Position Encoding）
RNN 在循环过程中，每个词按顺序输入，因此隐含地知道每个词的位置。
然而，Transformer一个序列中的所有词都是并行输入的。这是其相对于RNN 架构的主要优势；但同时也意味着位置信息会丢失，必须单独添加回来。

解码器堆栈和编码器堆栈各有一个位置编码层。位置编码的计算是独立于输入序列的，是固定值，只取决于序列的最大长度。
第一项是一个常数代码，表示第一个位置。
第二项是一个表示第二位置的常量代码。

3. 矩阵维度（Matrix Dimensions）
深度学习模型一次处理一批训练样本。嵌入层和位置编码层对一批序列样本的矩阵进行操作。嵌入层接受一个 (samples, sequence_length) 形状的二维单词ID矩阵，将每个单词ID编码成一个单词向量，其大小为 embedding_size，从而得到一个（samples, sequence_length, embedding_size) 形状的三维输出矩阵。位置编码使用的编码尺寸等于嵌入尺寸。所以它产生一个类似形状的矩阵，可以添加到嵌入矩阵中。

4. Encoder
编码器和解码器堆栈分别由几个（通常是 6 个）编码器和解码器组成，按顺序连接。
	堆栈中的第一个编码器从嵌入和位置编码中接收其输入。堆栈中的其他编码器从前一个编码器接收它们的输入。
	当前编码器接受上一个编码器的输入，并将其传入当前编码器的自注意力层。当前自注意力层的输出被传入前馈层，然后将其输出至下一个编码器。

5. Decoder
解码器的结构与编码器的结构非常类似，但有一些区别。

- 像编码器一样，堆栈中的第一个解码器从嵌入层（词嵌入+位置编码）中接受输入；堆栈中的其他解码器从上一个解码器接受输入。
- 在一个解码器内部，输入首先进入自注意力层，这一层的运行方式与编码器相应层的区别在于：
	训练过程中，每个时间步的输入，是直到当前时间步所对应的目标序列，而不仅是前一个时间步对应的目标序列(即输入的是step0-stepT-1，而非仅仅stepT-1）。
	推理过程中，每个时间步的输入，是直到当前时间步所产生的整个输出序列。
	解码器的上述功能主要是通过 mask 方法进行实现得的。
- 解码器与编码器的另一个不同在于，解码器有第二个注意层层，即编码器-解码器注意力层 Encoder-Decoder-attention 层。其工作方式与自注意力层类似，只是其输入来源有两处：位于其前的自注意力层及 E解码器堆栈的输出。
- Encoder-Decoder attention 的输出被传入前馈层，然后将其输出向上送至下一个Decoder。
- Decoder 中的每一个子层，即 Multi-Head-Self-Attention、Encoder-Decoder-attention 和 Feed-forward层，均由一个残差连接，并进行层规范化。

## 残差连接和层规范化
残差连接（residual connections）和层规范化（layer normalization）是为了解决深层神经网络训练中的梯度消失和梯度爆炸问题，并提高模型的稳定性和训练效果。

残差连接意味着在每个层中，模型在进行计算之前，将输入直接与输出相加。这样做的好处是让模型能够保留原始输入的信息，避免信息的丢失。在传统的神经网络中，经过多个层的处理后，输入的信息可能被逐渐稀释，导致梯度消失或爆炸。

层规范化是对每一层的输出进行归一化操作，使得输出数据在特征维度上具有相近的分布。它的作用是将不同样本在每个特征维度上进行比较和学习。通过层规范化，可以提高模型的稳定性和泛化能力。另外，层规范化还有助于解决梯度问题，使得网络在训练过程中更容易收敛。

总之，残差连接和层规范化都是为了解决深层神经网络训练中的梯度问题，提高模型的稳定性和训练效果。残差连接通过保留原始输入信息，减轻梯度消失和爆炸问题；层规范化通过将输出数据归一化，提高模型的稳定性和学习能力。

```python
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

## 编码器

下面的`EncoderBlock`类包含两个子层：多头自注意力和基于位置的前馈网络，这两个子层都使用了残差连接和紧随的层规范化。

```python
class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(  # 定义一个多头注意力模型
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)  # 定义一个添加和正则化模块
        self.ffn = PositionWiseFFN(  # 定义一个位置感知全连接层
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):  # 前向传播方法
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))  # 注意力机制
        return self.addnorm2(Y, self.ffn(Y))  # 位置感知全连接层
```

这个类定义了一个Transformer编码器块，它包含一个多头注意力模型和一个位置感知全连接层。在前向传播过程中，先使用注意力机制对输入数据进行处理，然后再经过一个位置感知全连接层，最后返回处理后的结果。注意，在每次处理之前都会使用AddNorm模块进行正则化和添加操作。

```python
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

## 解码器
Transformer解码器也是由多个相同的层组成。在`DecoderBlock`类中实现的每个层包含了三个子层：解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络。这些子层也都被残差连接和紧随的层规范化围绕。

在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；然而，在预测阶段，其输出序列的词元是逐个生成的。
因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数`dec_valid_lens`，以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算。
构建解码器中的单独模块：
```python
class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(  # 第一个自注意力模型
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)  # 添加和正则化模块
        self.attention2 = d2l.MultiHeadAttention(  # 编码器-解码器注意力模型
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(  # 位置感知全连接层
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):  # 前向传播方法
        enc_outputs, enc_valid_lens = state[0], state[1]  # 获取编码器输出和有效长度
        if state[2][self.i] is None:  # 如果当前解码块的输出未初始化
            key_values = X  # 则使用解码器输入作为键值对
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)  # 否则将上一步的输出与当前输入拼接起来
        state[2][self.i] = key_values  # 更新解码器状态
        if self.training:  # 如果处于训练阶段
            batch_size, num_steps, _ = X.shape  # 计算解码器的有效长度
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:  # 如果处于预测阶段
            dec_valid_lens = None  # 解码器的有效长度默认为None
        # 自注意力处理
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力处理
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        # 位置感知全连接层处理
        return self.addnorm3(Z, self.ffn(Z)), state
```

这个类定义了一个解码器块，它包含了两个自注意力模型（第一个用于自注意力，第二个用于编码器-解码器注意力），一个位置感知全连接层以及几个添加和正则化模块。在前向传播过程中，先通过第一个自注意力模型和添加和正则化模块处理输入数据，然后再通过编码器-解码器注意力模型处理，最后通过位置感知全连接层和添加和正则化模块处理。



随后构建完整的解码器：
```python
class TransformerDecoder(d2l.AttentionDecoder):

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers  # 解码器层数目
        self.embedding = nn.Embedding(  # 将词汇表映射成嵌入空间
            vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(  # 坐标编码器
            num_hiddens, dropout)
        self.blks = nn.Sequential()  # 创建一个包含多个DecoderBlock的子模块列表
        for i in range(num_layers):  # 每一层都加入一个DecoderBlock
            blk_name = "block" + str(i)
            self.blks.add_module(blk_name,
                            DecoderBlock(
                                key_size, query_size, value_size,
                                num_hiddens, norm_shape,
                                ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, i))

        self.dense = nn.Linear(  # 线性变换层，用于输出结果
            num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):  # 初始化解码器状态
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):  # 前向传播方法
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [
            [None] * len(self.blks) for _ in range(2)]  # 存储注意力权重
        for i, blk in enumerate(self.blks):  # 对每一层解码器块进行处理
            X, state = blk(X, state)  # 处理输入数据
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights  # 解码器自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights  # 编码器-解码器自注意力权重
        return self.dense(X), state  # 返回最终结果和更新后的状态

    @property
    def attention_weights(self):  # 属性访问器，用于获取注意力权重
        return self._attention_weights
```

它有一个字典形式的状态变量`_attention_weights`，用于存储每个块中的注意力权重信息。
其中`self.blks`是包含多个`DecoderBlock`的序列，每个`DecoderBlock`都有自己的`attention1`和`attention2`属性。`TransformerDecoder`有三个主要部分：embedding层、坐标编码层、DecoderBlock序列，以及一个密集层。
在构造函数中，创建了一个`nn.Sequential`对象`blks`来保存所有的`DecoderBlock`对象。
对于`forward`函数，首先将输入序列经过嵌入层和坐标编码，并存储每一步的关注权值信息到`_attention_weights`字典中。
最后，将整个序列经过所有blocks，并返回输出和更新后的状态。 `transformer_decoder.init_state`函数初始化解码器状态`state`为三个元素：第一个元素是编码器的输出，第二个元素是有效长度列表，第三个元素是`state`对象的副本。
`forward`函数将解码器输入传递给每一层DecoderBlock，然后将所有注意力权值信息存储到`_attention_weights`中。
# 总结
Transformer可以简单地理解为一个处理自然语言的模型。它的特点是可以同时处理整个输入，不受顺序限制。传统的模型可能只能一个一个地处理单词，但Transformer可以一次性整体地理解句子中的每个单词，并根据它们的重要性来调整处理方式。
它的原理就像是我们在读一篇文章时，不是一个一个单词地读，而是看着整篇文章，在自己的注意力下，会自动找出关键词、重要句子并理解它们所表达的意思。Transformer也是这样的原理，通过自动地计算单词之间的相关性来理解它们的意义。
这样的特点使得Transformer在处理自然语言任务时表现得很出色，比如翻译和生成对话。因为它能够全面地理解整个句子或文本，得到更好的语义关系和上下文信息。同时，它的计算也比较高效，可以并行处理，提高了效率。