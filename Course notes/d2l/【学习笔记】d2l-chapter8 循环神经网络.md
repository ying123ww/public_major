
# 前言
循环神经网络RNN：用于处理序列数据和时间序列数据。且具有一种循环的结构，使其能够在处理**序列数据（input）** 时具有**记忆能力**。

RNN 的主要**特点**包括：

1. 循环结构：RNN 的神经元之间存在循环连接，使信息可以在网络中循环传递。这种结构使得 RNN 能够处理不定长度的序列数据，如自然语言文本、音频信号、时间序列等。

2. 内部状态（Hidden State）：RNN 的每个时间步都有一个内部状态，该状态包含了之前时间步的信息。这个内部状态在处理序列数据时可以被更新和传递，允许网络记住过去的信息并影响未来的输出。

3. 参数共享：RNN 在不同时间步使用相同的权重参数，这意味着它们在处理不同时间步的输入时使用相同的模型，从而减少了网络的参数数量。

问题：**长程依赖**问题。
长期依赖问题（Long-Term Dependency Problem）是指在循环神经网络（RNN）等序列模型中，当处理具有**很长时间间隔的序列数据时，模型很难捕捉到序列中较早时间步的信息**，导致性能下降的现象。这个问题在传统的RNN中尤为显著，因为RNN的内部状态在处理长序列时会受到**梯度消失或梯度爆炸**等问题的影响，从而导致较早时间步的信息无法有效传递和保持。

改进：RNN 变体，包括**长短时记忆网络（LSTM）和门控循环单元（GRU）** 等。 

LSTM 和 GRU 引入了额外的门控机制，可以更有效地捕捉长期依赖性，从而提高了 RNN 在处理序列数据时的性能。它们具有更复杂的内部结构，能够控制信息的添加和遗忘，从而更好地处理各种序列数据任务，如自然语言处理、机器翻译、语音识别、时间序列预测等。


# 序列模型
根据前面的时间序列（input）预测当下$x_t$：
$$x_t\sim P(x_t\mid x_{t-1},\ldots,x_1)$$
预测方法如下：
## 自回归模型
解决如何有效估计$P(x_t\mid x_{t-1},\ldots,x_1)$：

1.自回归模型
只采用长度为$\tau$的数据，即$\begin{aligned}x_{t-1},\ldots,x_{t-\tau}\end{aligned}$,这样参数数量不变。

2.隐变量自回归模型
保留对过去观测的总结 $h_t$ ，同时更新预测 $\hat{x}_t$ 和总结 $h_t$
![在这里插入图片描述](https://img-blog.csdnimg.cn/0fb939408a1d4cc8a2d22b3e588eafa8.png#pic_center)

 - 利用 $\hat{x}_t=P(x_t\mid h_t)$ 估计 $x_t$
 - 利用公式 $h_t=g(h_{t-1},x_{t-1})$ 更新的模型
 
问题：训练数据哪里来? 使用历史观测来预测下了一个未来观测。但如果想预测一个序列呢？

合理假设：数据的动力学不变，固有规律不变，即序列**平稳**。

那么估计值为：$$\begin{aligned}P(x_1,\ldots,x_T)&=\prod_{t=1}^TP(x_t\mid x_{t-1},\ldots,x_1).\end{aligned}$$
***
推导：

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1})
$$

通过概率论中的链式法则来实现。

首先，回顾一下链式法则：

链式法则表明，对于任何事件序列 $A_1, A_2, \ldots, A_n$，其联合概率可以分解为条件概率的连乘形式：

$$
P(A_1, A_2, \ldots, A_n) = P(A_1)P(A_2 \mid A_1)P(A_3 \mid A_1, A_2) \ldots P(A_n \mid A_1, A_2, \ldots, A_{n-1})
$$

现在，将这个链式法则应用到文本序列上，其中 $A_i$表示在时间步 $i$ 处观察到的词元，也就是 $x_i$。因此，可以得到以下推导：

$$
\begin{align*}
P(x_1, x_2, \ldots, x_T) &= P(x_1)P(x_2 \mid x_1)P(x_3 \mid x_1, x_2) \ldots P(x_T \mid x_1, x_2, \ldots, x_{T-1})
\end{align*}
$$

它基于联合概率分布的定义和链式法则，将文本序列的联合概率分解为每个词元的条件概率的连乘形式。这个分解允许我们对文本序列的概率建模，并通过估计每个条件概率来预测或生成文本。

注：处理对象离散（单词）：分类器。处理对象连续（数字）：回归模型。
***
## 马尔可夫模型

> 马尔可夫条件：在一个随机过程中，当前状态的概率分布仅依赖于**前一个状态**，而不依赖于更早的状态。

在自回归中，提到了采用长度为$\tau$的数据，即$\begin{aligned}x_{t-1},\ldots,x_{t-\tau}\end{aligned}$,代替$\begin{aligned}x_{t-1},\ldots,x_{1}\end{aligned}$.
如果假设 $\tau=1$ 则用$x_{t-1}$,代替$\begin{aligned}x_{t-1},\ldots,x_{1}\end{aligned}$.
则上述公式会变为
$$\begin{aligned}P(x_1,\ldots,x_T)&=\prod_{t=1}^TP(x_t\mid x_{t-1})\text{当}P(x_1\mid x_0)=P(x_1).\end{aligned},$$
即一阶马尔可夫模型。

这样的话就非常容易从动态规划入手，进行不断地预测。

## 序列模型代码
由于上面都讲述了理论的预测，于是我们打算实践，看看预测效果准不准。我们将序列数据变成“序列-标签”对的形式。大致$\tau$个时间布对应后面一个时间步。即**将数据映射为数据对$y_t = x_t$和$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$。** 然后用前600个在神经网络（MLP）去训练。随后用k步预测看看效果如何（全部数据为1000个）并分析。

```python
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32) #离散序列
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) #增加扰动



#设置数据对
#设置好特征-标签对
tau = 4 #设置序列长度tau
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))
#.reshape((-1, 1))：这是对切片后的数组进行形状重塑（reshape）操作。.reshape() 方法允许你改变数组的形状。在这里，(-1, 1) 表示将数组重新塑造成一个二维数组，其中每行有一个元素。-1 在这里表示根据原始数据的大小来自动确定行数，而 1 表示每行只有一个元素。因此，这个操作将原始切片后的一维数组转换为一个列向量。



#用前n个数据训练
#当 batch_size 和 n_train 相等时，采用的是全批量（batch）训练方式，也称为批量梯度下降。每次迭代都使用整个训练数据集来更新模型的参数
#BATCH_SIZE:即一次训练所抓取的数据样本数量；
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)




#训练的神经网络
# 初始化网络权重的函数
def init_weights(m):
    # 检查当前模块是否为线性层（全连接层）
    if type(m) == nn.Linear:
        # 对线性层的权重进行初始化，使用 Xavier 初始化方法
        nn.init.xavier_uniform_(m.weight)

# 定义一个简单的多层感知机（MLP）神经网络
def get_net():
    # 使用 nn.Sequential 定义一个序列模型
    net = nn.Sequential(
        nn.Linear(4, 10),  # 输入层到隐藏层1，输入特征维度为4，隐藏层维度为10
        nn.ReLU(),         # 隐藏层1的激活函数为 ReLU
        nn.Linear(10, 1)   # 隐藏层2到输出层，隐藏层维度为10，输出层维度为1
    )
    
    # 调用初始化函数，对网络的权重进行初始化
    net.apply(init_weights)
    
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')



#训练过程
def train(net, train_iter, loss, epochs, lr):
    # 创建Adam优化器，用于更新神经网络参数
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        # 遍历训练数据集
        for X, y in train_iter:
            # 清零梯度，以准备接收新的梯度值
            trainer.zero_grad()
            # 计算网络的预测结果
            l = loss(net(X), y)
            # 计算损失函数对参数的梯度
            l.sum().backward()
            # 使用优化器来更新模型参数
            trainer.step()
        # 打印每个epoch的损失值
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

# 创建神经网络模型
net = get_net()
# 调用训练函数，训练模型
train(net, train_iter, loss, 5, 0.01)



#试试单步预测
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))


#试试多步预测（使用自己预测出来的数据继续预测），而不是用原始数据预测
# 创建一个初始多步预测序列multistep_preds，长度为T
multistep_preds = torch.zeros(T)

# 将前n_train+tau个时间步的原始数据复制到多步预测序列中
multistep_preds[:n_train + tau] = x[:n_train + tau]

# 遍历从n_train+tau到T的时间步，进行多步预测
for i in range(n_train + tau, T):
    # 使用神经网络模型net对过去tau个时间步的数据进行预测，
    # 并将预测结果放入multistep_preds的相应位置
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/9f6d30db29244bca8535c58439b3f60c.png)
可以看出多步预测的结果比单步要差很多，这是因为误差是会累计的。
随后我们尝试k步预测，发现当我们想要预测更远的

```python
#k步预测
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/8fee89d34c2240c2bd426b4139cb5b1f.png)
## 序列数据处理——文本预处理
> 词元（Token）是文本处理中的基本单位，通常指的是文本中的一个单词、一个字符或者其他可以被拆分并独立处理的最小单位。在自然语言处理（NLP）和文本处理中，文本通常会被分解成词元的序列，以便进行分析、建模、处理或其他操作。
1.  将文本作为字符串加载到内存中。
2. 将字符串拆分为词元（如单词和字符）。
3. 建立一个词表，将拆分的词元映射到数字索引。
4. 将文本转换为数字索引序列。

```python
import collections
import re
from d2l import torch as d2l


#文本获取
# 在d2l库的DATA_HUB中添加时间机器数据集的下载链接和校验和
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    # 打开时间机器数据集文件并读取所有行
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 对每一行文本进行处理，去除非字母字符，转换为小写，并去除首尾空白字符
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 调用read_time_machine函数加载时间机器数据集
lines = read_time_machine()

# 打印文本总行数和前两行文本内容的示例
print(f'# 文本总行数: {len(lines)}')
print(lines[0])  # 打印第一行文本内容
print(lines[10]) # 打印第十一行文本内容




#词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        # 如果token参数为'word'，则使用空格分割每一行文本，将文本行拆分为单词列表
        return [line.split() for line in lines]
    elif token == 'char':
        # 如果token参数为'char'，则将每一行文本拆分为字符列表
        return [list(line) for line in lines]
    else:
        # 如果token参数既不是'word'也不是'char'，则输出错误消息
        print('错误：未知词元类型：' + token)

# 调用tokenize函数，将文本行拆分为单词列表
tokens = tokenize(lines)

# 打印前11行文本的词元化结果
for i in range(11):
    print(tokens[i])



#转化为词表：
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 统计词元频率并按频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 未知词元的索引为0，将保留词元添加到词表中
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # 将高频词元添加到词表中，如果频率低于min_freq则停止添加
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 获取词元的索引，如果词元不存在，则返回未知词元的索引
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # 根据索引获取词元
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        # 返回词元频率信息
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens可以是1D列表或2D列表，将词元列表展平成一个列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 创建一个Vocab对象vocab，基于词元列表tokens构建词表
vocab = Vocab(tokens)

# 打印词表中前10个词元的索引和词元对应关系
print(list(vocab.token_to_idx.items())[:10])



# 定义一个函数load_corpus_time_machine，用于加载时光机器数据集的词元索引列表和词表
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    # 读取时光机器数据集的文本行
    lines = read_time_machine()
    
    # 将文本行拆分为字符词元，并创建词表vocab
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    
    # 将所有词元展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    
    # 如果指定了max_tokens，限制词元索引列表的长度
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    
    return corpus, vocab

# 调用load_corpus_time_machine函数加载时光机器数据集的词元索引列表和词表
corpus, vocab = load_corpus_time_machine()

# 打印词元索引列表的长度和词表的长度
len(corpus), len(vocab)

```
于是文本序列：
假设长度为$T$的文本序列中的词元依次为$x_1, x_2, \ldots, x_T$。
于是，$x_t$（$1 \leq t \leq T$）可以被认为是文本序列在时间步$t$处的观测或标签。


## 从序列模型到语言模型（过渡）

1. **序列模型**：
   - 序列模型是一种广泛用于处理序列数据的机器学习模型，它的**输入和输出都是序列**。
   - 序列模型可以是有监督学习模型，用于序列预测任务，如序列到序列翻译、时间序列预测等。它们也可以是无监督学习模型，用于序列数据的降维、聚类等任务。
   - 序列模型的目标是捕捉序列数据中的模式和依赖关系，以便进行有意义的预测或分析。

2. **语言模型**：
   - **语言模型是一种特殊的序列模型**，它专门用于自然语言处理（NLP）领域，对语言文本进行建模。
   - 语言模型的主要任务是预测给定上下文中的下一个单词或字符。它可以用于文本生成、自动纠错、文本分类等任务。
   - 语言模型的**输入通常是一个序列（例如，前面的单词），输出是一个条件概率分布**，表示下一个可能的单词或字符。（预测）

总的来说，语言模型是序列模型的一个特例，它针对文本数据的序列建模，而序列模型可以应用于更广泛的序列数据，如时间序列、音频信号、图像序列等。
# 语言模型



*语言模型*（language model）的目标是**估计序列的联合概率**
$$P(x_1, x_2, \ldots, x_T)，$$
通过估计文本序列的联合概率，语言模型可以量化文本中不同单词或字符之间的关联程度，从而更好地理解和生成自然语言文本。



首先联合概率可以被拆分条件概率分布：
$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

>上文已推导出。这个等式成立是因为它基于条件独立性假设（Conditional Independence Assumption）。这个假设表明，在给定前面的所有词元的情况下，当前词元的出现是相互独立的。换句话说，文本序列中每个词元的生成都只依赖于其前面的词元，而与其他词元无关。
>这个假设在语言建模中是一个常见的近似，它使问题变得更加可管理。

案例：

包含了四个单词的一个文本序列的概率是：
$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$
1.估计P(deep):（稍稍不太精确的）方法是统计单词“deep”在数据集中的出现次数，
然后将其除以整个语料库中的单词总数。
2.估计$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$其中$n(x)$和$n(x, x')$分别是单个单词和连续单词对的出现次数。
这里假设采用的是频率统计法。

>问题：对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。而对于三个或者更多的单词组合，情况会变得更糟。许多合理的三个单词组合可能是存在的，但是在数据集中却找不到。

需解决：将这些单词组合指定为非零计数。

方法：*拉普拉斯平滑*（Laplace smoothing）：在所有计数中添加一个小常量。用$n$表示训练集中的单词总数，用$m$表示唯一单词的数量。

$$
\begin{aligned}
    \hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
    \hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
    \hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}
$$

其中，$\epsilon_1,\epsilon_2$和$\epsilon_3$是超参数。
以$\epsilon_1$为例：当$\epsilon_1 = 0$时，不应用平滑；
当$\epsilon_1$接近正无穷大时，$\hat{P}(x)$接近均匀概率分布$1/m$。

***





## 传统方法
### 马尔可夫模型和n元语法

引入：由于上文的建模需要联合概率，而联合概率可以被拆分条件概率分布，但是针对实际求解问题，我们的条件概率p(Xt)或许可以不用和前面所有时刻都相关，或许我们可以用马尔可夫模型去缩短上下文依赖性的长度。


对于马尔可夫模型，**不同的阶数指的是模型考虑的上下文依赖性的长度或距离**。马尔可夫模型的一阶到$n$阶模型分别考虑了不同长度的上下文信息。阶数越高，对应的依赖关系就越长。

- 一阶马尔可夫模型：一阶马尔可夫模型假设每个状态（或观测）只依赖于前一个状态，即$P(X_t \mid X_1, X_2, \ldots, X_{t-1}) = P(X_t \mid X_{t-1})$。这意味着模型只考虑最近一个时刻的状态，不考虑更早时刻的状态。

- 二阶马尔可夫模型：二阶马尔可夫模型考虑每个状态依赖于前两个状态，即$P(X_t \mid X_1, X_2, \ldots, X_{t-1}) = P(X_t \mid X_{t-1}, X_{t-2})$。这使得模型能够考虑到更长的历史信息，因此具有更大的上下文依赖性。

- 三阶马尔可夫模型：三阶马尔可夫模型考虑每个状态依赖于前三个状态，以此类推。

这种性质推导出了语法公式：
$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$
涉及一个、两个和三个变量的概率公式分别被称为*一元语法*（unigram）、*二元语法*（bigram）和*三元语法*（trigram）模型。
> 一阶马尔可夫模型和二元语法的关系
> 可以说二元语法是一阶马尔可夫模型的一个特例，它在自然语言处理中常用于建模文本数据中的词序关系。而一阶马尔可夫模型可以用于更广泛的随机序列建模问题，不仅局限于文本数据。

对于二元语法的训练过程来说：

具体步骤包括：
 1. 统计训练数据中每个词的出现频率以及词对（bigram）的共现频率。
 2. 计算条件概率$P(w_t \mid w_{t-1})$，其中$w_t$表示当前词，$w_{t-1}$表示前一个词。这可以通过简单地将词对的共现频率除以前一个词的出现频率来计算。

所以对于数据的统计分析

### 词元统计
打印前$10$个最常用的（频率最高的）单词。
在这段代码中，"token" 是指文本中的最小语义单元，通常是一个单词或一个标点符号。在自然语言处理中，文本通常会被分割成单独的 token 来进行处理。

```python
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

1. `tokens = d2l.tokenize(d2l.read_time_machine())`：这行代码首先通过`d2l.read_time_machine()`读取了一段文本数据，然后使用`d2l.tokenize`函数将文本分割成 token，即将文本拆分成单词或标点符号等语言学单位。`tokens` 是一个包含所有 token 的列表。

2. `corpus = [token for line in tokens for token in line]`：这行代码将所有的 token 汇总到一个名为 `corpus` 的列表中，它将文本中的每行的 token 连接成一个整体的 token 列表。这个操作的目的是为了构建一个包含整个文本数据集的 token 列表。

3. `vocab = d2l.Vocab(corpus)`：这行代码使用 `d2l.Vocab` 构建了一个词汇表（vocabulary）。词汇表是一个包含文本数据中所有不同 token 的集合，它记录了每个 token 出现的频率等信息。

4. `vocab.token_freqs[:10]`：这行代码打印出词汇表中前 10 个 token 的频率信息，即它们在文本数据中出现的次数。

打印结果：

('the', 2261),
 ('i', 1267),
 ('and', 1245),
 ('of', 1155),
 ('a', 816),
 ('to', 695),
 ('was', 552),
 ('in', 541),
 ('that', 443),
 ('my', 440)


所以，整个代码的目的是将文本数据中的 token 进行处理，并构建一个词汇表，以便后续的自然语言处理任务中使用。


画出词频图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/6689973e40d04489b2776d0b8862848f.png)
将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线。

***
这意味着单词的频率满足**齐普夫定律**（Zipf's law），
即第$i$个最常用单词的频率$n_i$为：

$$n_i \propto \frac{1}{i^\alpha},$$


等价于

$$\log n_i = -\alpha \log i + c,$$

其中$\alpha$是刻画分布的指数，$c$是常数。

>什么是齐普夫定律？

齐普夫定律的核心观点是：在大多数自然语言中，某个单词的使用频率与其在单词频率排名表中的排名成反比。具体来说，如果将词汇表中的单词按照它们在文本中的出现频率从高到低排列，那么排名第二的单词的使用频率将约为排名第一单词的频率的一半，排名第三的单词的频率将约为排名第一单词的频率的三分之一，以此类推。这种规律表现为一个幂律分布，通常以Zipf分布来描述。
它表明一小部分单词在文本中频繁出现，而大多数单词出现频率很低。这些高频率的单词通常是常用词汇（如"the"、"and"、"in"等），而低频率的单词则可能是生僻词汇或专业术语。

***

继续对比二元和三元：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8abe4c3c24aa4fa0b32b609cadb8f715.png)
1. 除了一元语法词，单词序列似乎也遵循齐普夫定律。
2. 词表中$n$元组的数量并没有那么大。
3. 很多$n$元组很少出现，这使得拉普拉斯平滑非常不适合语言建模。作为代替，我们将使用基于**深度学习**的模型。

## 深度学习方法
### 处理长序列数据
在之前的行为：
当序列变得太长而不能被模型一次性全部处理时则拆分。
神经网络模型中：
一次处理具有预定义长度（例如$n$个时间步）的一个小批量序列。
现在的问题是如何 **随机生成一个小批量数据的特征和标签以供读取**

假设网络一次只处理具有n个时间步的子序列，但是我们可以选择任意偏移量来指定初始位置。
![[Pasted image 20231105134637.png]]
图上不同的偏移量会导致不同的子序列。

1. 要么偏移量就随机，因为只选择一个偏移量，那么训练网络的子序列的覆盖范围就是有限的。
2. 要么直接顺序分区。

总结：子序列提取有两种方法：*随机采样*（random sampling）和*顺序分区*（sequential partitioning）策略

#### 随机采样

- 特点：随机采样是一种从长序列中随机选择样本的方法，每个样本的选择是独立的，没有考虑样本之间的时间关系。
- 应用场景：随机采样适用于需要随机化数据顺序的任务，以避免模型过度拟合特定的时间顺序。它通常用于文本数据的批量训练，其中可以随机选择不同的句子或段落作为训练样本。
- 优点：随机采样简单，适用于计算成本较低的情况下，且可以**更好地避免时间相关性的影响**。
- 注意事项：随机采样可能导致丢失时间序列的信息，因此在某些任务中可能不适用。

```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

#### 顺序分区

- 特点：顺序分区是一种将长序列划分为连续的子序列或分区的方法，每个分区都包含一系列连续的时间步。分区之间通常没有重叠。
- 应用场景：顺序分区通常用于序列模型的交叉验证或模型评估中。它可以确保模型在每个分区中都有机会观察整个时间序列的不同部分，从而更好地评估模型的性能。
- 优点：顺序分区保留了时间序列的结构，适用于**需要考虑时间相关性的任务**，如时间序列预测或信号处理。
- 注意事项：顺序分区可能导致**较大的计算成本**，特别是在较长的序列上，因为每个分区都需要进行模型训练和评估。

```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

于是将上述两个采样都包装到一个类中：

```python
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

定义一个函数：返回迭代器和词表
```python
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```
***
#### Q&A
1. 假设训练数据集中有$100,000$个单词。一个四元语法需要存储多少个词频和相邻多词频率？
   - 对于一个四元语法（four-gram），需要存储的词频数量取决于训练数据集中的唯一单词数。假设有$V$个唯一单词，那么需要存储的四元词频（四元组的频率）的数量是$V^4$。此外，还需要存储相邻多词频率，这将是$V^3$个三元词频（三元组的频率）和$V^2$个二元词频（二元组的频率）。
   
2. 我们如何对一系列对话建模？
   - 对一系列对话进行建模通常涉及将每个对话分解为单独的文本序列，其中每个序列可以表示为一系列词元或符号。这些文本序列可以组成一个大型的文本语料库，然后使用语言模型来建模。语言模型可以用于生成对话、回答问题、识别情感等任务。在对话建模中，上下文处理和对话历史的管理也是关键问题。

3. 一元语法、二元语法和三元语法的齐普夫定律的指数是不一样的，能设法估计么？
   - 可以通过对训练数据集中的词元进行统计来估计一元、二元和三元语法的频率分布。一元语法的频率分布可以直接从单个词元的出现次数估计得出。对于二元语法，可以计算相邻词元对的频率，以及给定前一个词元的情况下下一个词元的频率。对于三元语法，可以计算三个相邻词元的频率，以及给定前两个词元的情况下第三个词元的频率。这些估计可以用于分析文本数据中的词元关系和频率分布。

4. 想一想读取长序列数据的其他方法？
   - 读取长序列数据的其他方法包括滑动窗口采样、分块采样、滑动窗口分块采样等。这些方法可以帮助处理长序列数据，减小计算复杂度，并确保模型能够观察整个序列的不同部分。

5. 考虑一下我们用于读取长序列的随机偏移量。
   - 为什么随机偏移量是个好主意？
     - 随机偏移量可以引入多样性，确保模型在不同位置观察数据，从而更好地泛化到不同的情况。
   - 它真的会在文档的序列上实现完美的均匀分布吗？
     - 不一定，随机偏移量的分布可能在某些位置更密集，而在其他位置更稀疏，具体取决于随机性。
   - 要怎么做才能使分布更均匀？
     - 要使分布更均匀，可以尝试不同的随机偏移量生成方法，例如均匀分布或高斯分布。还可以调整随机种子以改变随机性，以获得更均匀的分布。

6. 如果我们希望一个序列样本是一个完整的句子，那么这在小批量抽样中会带来怎样的问题？如何解决？
   - 如果希望一个序列样本是一个完整的句子，并且句子跨越多个小批量，可能会导致问题，因为模型可能在小批量之间丢失句子的上下文信息。
   - 解决方法包括：
     - 使用**截断和填充**：将句子截断或填充到相同的长度，以确保它们适合于小批量处理，但可能会丢失部分信息。
     - 使用**动态填充**：在小批量中动态选择句子长度，以最大程度地减小填充的数量。
     - 使用**束搜索**：在生成任务中，可以使用束搜索来考虑多个候选句子，以获得更好的生成结果。
     - 使用**分段采样**：将长句子分成短段落或子句进行处理，然后组合结果。
   - 选择合适的方法取决于具体的任务和模型需求。

***

### 循环神经网络
#### 背景
由于$n$元语法模型中，单词$x_t$在时间步$t$的条件概率仅取决于前面$n-1$个单词，即需计算$P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$，如果为了准确让n无限增大，词表$\mathcal{V}$需要存储$|\mathcal{V}|^n$个数字，内存较大。不如直接将前面的影响全部归结到该时刻的隐状态$h_{t-1}$上。
使用隐变量模型：
$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

其中$h_{t-1}$是*隐状态*（hidden state），存储了到时间步$t-1$的序列信息。
通常，我们可以基于当前输入$x_{t}$和先前隐状态$h_{t-1}$
来计算时间步$t$处的任何时间的隐状态：
$$h_t = f(x_{t}, h_{t-1}).$$

***
>隐藏层和隐状态：隐藏层是在从输入到输出的路径上（以观测角度来理解）的隐藏的层，而隐状态则是在给定步骤所做的任何事情（以技术角度来定义）的*输入*.并且这些状态只能通过先前时间步的数据来计算.


1. **隐藏层（Hidden Layer）**：
   - 隐藏层是RNN模型中的一个神经网络层，通常包含多个神经元（也称为隐藏单元）。
   - 隐藏层的主要作用是对输入数据进行转换和特征提取。每个时间步，输入数据（通常是当前时间步的输入序列和上一个时间步的隐状态）经过隐藏层的计算，生成一个新的输出，用于传递到下一个时间步。
   - 隐藏层的参数（权重和偏差）在训练过程中学习，以适应特定任务的数据。

2. **隐状态（Hidden State）**：
   - 隐状态是RNN模型中的一个重要概念，它表示模型在处理序列数据时在不同时间步的内部状态或记忆。
   - 隐状态在RNN中被视为一个向量，其维度通常由隐藏层的神经元数量决定。在每个时间步，RNN会更新隐状态，并将其传递到下一个时间步，以便模型可以记住之前时间步的信息。
   - 隐状态包含了过去时间步的信息，因此具有一定的"记忆"能力，它可以捕捉到序列中的长期依赖关系。

***

#### 从MLP到RNN

MLP：
隐藏层的输出$\mathbf{H} \in \mathbb{R}^{n \times h}$通过下式计算：
$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
隐藏层权重参数为$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$，偏置参数为$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，以及隐藏单元的数目为$h$。
输出层：$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$
其中，$\mathbf{O} \in \mathbb{R}^{n \times q}$是输出变量，$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$是权重参数，$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的偏置参数。


RNN：
与MLP不同的是，我们在这里保存了前一个时间步的隐藏变量$\mathbf{H}_{t-1}$，并引入了一个新的权重参数$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，来描述如何在当前时间步中使用前一个时间步的隐藏变量。

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$

对于时间步$t$，输出层的输出类似于多层感知机中的计算：
$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

![](https://img-blog.csdnimg.cn/4d7716a0c32341979e355fcca4dae5ce.png)
***
证明：
>隐状态中$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$的计算，相当于$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的拼接与$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的拼接的矩阵乘法。

1.数学证明
要证明$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$的计算等价于$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的拼接与$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的拼接的矩阵乘法，我们可以通过矩阵运算来展示它们之间的等式。

首先，假设$\mathbf{X}_t$是维度为$n$的列向量，$\mathbf{H}_{t-1}$也是维度为$n$的列向量，$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$分别是维度为$n \times m$和$n \times n$的矩阵。我们可以表示$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$如下：

$$
\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh} = [\mathbf{X}_t, \mathbf{H}_{t-1}] \begin{bmatrix} \mathbf{W}_{xh} \\ \mathbf{W}_{hh} \end{bmatrix}
$$

这里，$[\mathbf{X}_t, \mathbf{H}_{t-1}]$表示将$\mathbf{X}_t$和$\mathbf{H}_{t-1}$在列维度上拼接起来，形成一个维度为$2n$的列向量。$\begin{bmatrix} \mathbf{W}_{xh} \\ \mathbf{W}_{hh} \end{bmatrix}$表示将$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$在行维度上拼接起来，形成一个维度为$(n+m) \times n$的矩阵。

现在，我们可以使用矩阵乘法的定义来展开上述表达式：

$$
[\mathbf{X}_t, \mathbf{H}_{t-1}] \begin{bmatrix} \mathbf{W}_{xh} \\ \mathbf{W}_{hh} \end{bmatrix} = [\mathbf{X}_t, \mathbf{H}_{t-1}] \begin{bmatrix} \mathbf{W}_{xh} \cdot \mathbf{W}_{hh} \end{bmatrix}
$$

在这个步骤中，我们实际上是将$\begin{bmatrix} \mathbf{W}_{xh} \\ \mathbf{W}_{hh} \end{bmatrix}$看作一个整体矩阵，然后将它与$[\mathbf{X}_t, \mathbf{H}_{t-1}]$相乘。这就等价于将$\mathbf{X}_t$和$\mathbf{H}_{t-1}$拼接在一起，并将其与$\mathbf{W}_{xh} \cdot \mathbf{W}_{hh}$相乘，这正是想要证明的。

最终，我们得到了等式$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh} = [\mathbf{X}_t, \mathbf{H}_{t-1}] \begin{bmatrix} \mathbf{W}_{xh} \cdot \mathbf{W}_{hh} \end{bmatrix}$，证明了这两者之间的等价性。

代码证明：

```python
import torch
from d2l import torch as d2l
```
相加：
```python
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
```
相乘：沿列（轴1）拼接矩阵X和H， 沿行（轴0）拼接矩阵W_xh和W_hh。 这两个拼接分别产生形状 和形状 的矩阵。 再将这两个拼接的矩阵相乘。
```python
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
```
结果相同。


#### 反向传播




RNN的数学公式：
$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$
![](https://img-blog.csdnimg.cn/4d7716a0c32341979e355fcca4dae5ce.png)



##### 1.通过时间反向传播（BPTT）

- 参数设置：
时间步$t$的隐状态表示为$h_t$，输入表示为$x_t$，输出表示为$o_t$。
由于隐状态中$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$的计算，相当于$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的拼接与$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的拼接的矩阵乘法。我们把隐藏层的权重统一为$w_h$。
$w_h$和$w_o$来表示隐藏层和输出层的权重。
$f$和$g$分别是隐藏层和输出层的变换。

- 公式：
$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$

- 前向传播：

有了$x_{t}$,于是可以计算 $h_t$ ,于是计算 $o_t$ 。一个时间步的遍历三元组$(x_t, h_t, o_t)$,于是可以组成链：

$$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$$
传播到最后可以算出Loss function：
$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$
即所有$T$个时间步内评估输出$o_t$和对应的标签$y_t$之间的差异。

注意：![[Pasted image 20231105145410.png]]RNN的特点是共享参数（W，U，V），可以看到这里无论是W、U、V在每个时间步上都是同一个，因此若说CNN是空间上的权值共享，那么RNN就是时间步上的共享。
并且对于左图来说，并不代表只有一个时间步，展开之后应有$L$个时间结构。以NLP处理为例，将一个句子进行分词，固定单词个数为$L$，则每个词的词向量则对应信号输入 $x_t$.



- 反向传播：

当计算目标函数$L$关于参数$w_h$的梯度时：
$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$

比较难以计算的是第三项$\partial h_t/\partial w_h$，$h_t$既依赖于$h_{t-1}$又依赖于$w_h$，其中$h_{t-1}$的计算也依赖于$w_h$。($h_{t-1}$的计算也依赖于$w_h$这个很困惑，后来了解到由于w是参数共享的，所以这个公式改成$h_{t-1}$的时候，依旧是这个$w_h$,所以就会造成循环嵌套的感觉)。

对于第三项，因为其中$h_t$是一个复合函数，依赖于多个变量，包括$h_{t-1}$和$w_h$，所以用链式法则产生：
$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$


基于下列公式替换成$a_t$、$b_t$和$c_t$：

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

可以转变为$$a_{t}=b_{t}+c_{t}a_{t-1}$$
这是一个递推公式，推导为a的一般公式，当$t\geq 1$，$a_{0}=0$时，就很容易得出：

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$

这时候再把原来的式子代回，就可以得到：

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$

计算方法：
![[Pasted image 20231105174025.png]]
上图从上到下分别是随机截断、常规截断、完整计算。

1. 完整计算。
	但是当$t$ 很大的时候，这个链就非常的长，所以我们要用其他方法。一般不可取。

2. 截断时间步。
	当模型比较侧重短期影响而不是长期影响的时候，就可以在$\tau$步后截断求和计算。

3. 随机截断。
	用一个随机变量替换$\partial h_t/\partial w_h$。随机变量是通过使用序列$\xi_t$来实现的，序列预定义了$0 \leq \pi_t \leq 1$，其中$P(\xi_t = 0) = 1-\pi_t$且$P(\xi_t = \pi_t^{-1}) = \pi_t$，因此$E[\xi_t] = 1$。我们使用它来替换梯度$\partial h_t/\partial w_h$得到：

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

从$\xi_t$的定义中推导出来$E[z_t] = \partial h_t/\partial w_h$。
每当$\xi_t = 0$时，递归计算终止在这个$t$时间步。
这导致了不同长度序列的加权和，其中长序列出现的很少，
所以将适当地加大权重。

但是随机截断理论可行，实际效果不好。原因有三：1.在对过去若干个时间步经过反向传播后，观测结果足以捕获实际的依赖关系。 2.增加的方差抵消了时间步数越多梯度越精确的事实。3.我们真正想要的是只有**短范围交互的模型**。
因此，模型需要的正是截断的通过时间反向传播方法所具备的轻度正则化效果。

##### 2.反向传播梯度细节

详情可以参考
https://zhuanlan.zhihu.com/p/61472450
讲得非常不错。

1. 假设：
考虑一个没有偏置参数的循环神经网络。
其在隐藏层中的激活函数使用恒等映射（$\phi(x)=x$）。

2. 具体公式：

对于时间步$t$，设单个样本的输入及其对应的标签分别为$\mathbf{x}_t \in \mathbb{R}^d$和$y_t$。
计算隐状态$\mathbf{h}_t \in \mathbb{R}^h$和输出$\mathbf{o}_t \in \mathbb{R}^q$的方式为：

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

其中权重参数为$\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。
用$l(\mathbf{o}_t, y_t)$表示时间步$t$处（即从序列开始起的超过$T$个时间步）的损失函数，
总体损失是：

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

三个时间步的计算图如下所示：
![[Pasted image 20231105175150.png]]

模型参数是$\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$和$\mathbf{W}_{qh}$。
通常，训练该模型需要对这些参数进行梯度计算：$\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$、$\partial L/\partial \mathbf{W}_{qh}$。
于是沿箭头的相反方向遍历计算图，依次计算和存储梯度。

1. 目标函数关于输出的微分$\frac{\partial L}{\partial \mathbf{o}_t}$：
$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$

2. 目标函数对于输出层的梯度$\frac{\partial L}{\partial \mathbf{W}_{qh}}$：
$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$
3. 目标函数对最后时间步$T$隐状态的梯度$\frac{\partial L}{\partial \mathbf{h}_T}$：

在最后的时间步$T$，目标函数$L$仅通过$\mathbf{o}_T$依赖于隐状态$\mathbf{h}_T$。因此，我们通过使用链式法则可以很容易地得到梯度$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$：

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$


4. 目标函数对其他时间步$t$隐状态的梯度$\frac{\partial L}{\partial \mathbf{h}_t}$：

当目标函数$L$通过$\mathbf{h}_{t+1}$和$\mathbf{o}_t$依赖$\mathbf{h}_t$时，对任意时间步$t < T$来说都变得更加棘手。
根据链式法则，隐状态的梯度$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$在任何时间步骤$t < T$时都可以递归地计算为：

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$


因为里面有$h_t$和$h_{t+1}$，所以我们整合一下，对于任何时间步$1 \leq t \leq T$展开递归计算得

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$

出现长序列模型的一些关键问题：
它陷入到$\mathbf{W}_{hh}^\top$的潜在的非常大的幂。
如果$\mathbf{W}_{hh}^\top$<1则会导致**梯度消失**，如果$\mathbf{W}_{hh}^\top$>1,则会造成**梯度爆炸**。


解决方法：
- **梯度截断**：截断时间步长的尺寸。避免它的幂太大。
- **梯度裁剪（Gradient Clipping）：** 这是应对梯度爆炸问题的一种方法。它涉及监测梯度的大小，并在梯度的大小超过某个阈值时对梯度进行缩放，以防止权重更新变得太大。
- **长短时记忆网络（LSTM）：** LSTM是一种特殊类型的RNN，设计用于捕捉长距离依赖关系。它使用门控机制来控制信息的流动，有助于减轻梯度消失问题。
- **权重初始化：** 合适的权重初始化方法，如Xavier初始化，可以有助于减轻梯度问题。
- **注意力机制：** 注意力机制允许模型有选择地关注输入序列的不同部分，有助于缓解长序列的问题，减少梯度传播的距离。

5. 计算参数梯度$\frac{\partial L}{\partial \mathbf{W}_{hx}}$和$\frac{\partial L}{\partial \mathbf{W}_{hh}}$
最后，目标函数$L$通过隐状态$\mathbf{h}_1, \ldots, \mathbf{h}_T$依赖于隐藏层中的模型参数$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$。为了计算有关这些参数的梯度$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$和$\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，我们应用链式规则得：

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

其中$\partial L/\partial \mathbf{h}_t$是前文递归计算得到的，是影响数值稳定性的关键量。所以梯度消失和梯度爆炸问题值得注意。

训练循环神经网络交替使用前向传播和通过时间反向传播。通过时间反向传播依次计算并存储上述梯度。具体而言，存储的中间值会被重复使用，以避免重复计算，例如存储$\partial L/\partial \mathbf{h}_t$，以便在计算$\partial L / \partial \mathbf{W}_{hx}$和$\partial L / \partial \mathbf{W}_{hh}$时使用。

### 质量度量指标——困惑度
***
为什么不选择似然概率去计算？：
在文本生成任务中，模型的目标是生成与观测数据（文本序列）相似的文本。因此，有时候我们使用序列的似然概率来度量文本模型的质量。具体来说，我们可以考虑以下两个方面：

1. **模型拟合度**：似然概率度量了模型生成观测文本序列的可能性。一个高似然概率表示模型生成观测文本的概率较高，因此模型在拟合数据方面较好。在文本生成任务中，我们希望模型生成的文本与实际观测到的文本尽可能相似，因此最大化似然概率有助于模型生成合适的文本。

2. **参数估计**：似然概率通常用于参数估计，如最大似然估计（Maximum Likelihood Estimation，MLE）或最大后验估计（Maximum A Posteriori Estimation，MAP）。通过最大化似然概率，我们可以找到能够最好地解释观测文本的模型参数。这些参数包括模型中的权重、偏差等，它们决定了文本生成模型的行为。

然而，在通过序列的似然概率来度量文本模型质量时，较短的序列比较长的序列更有可能出现的现象可能源于以下原因：

1. **与序列长度相关的概率问题**：似然概率的计算涉及到条件概率的乘积。较长的序列具有更多的单词或标记，因此整个序列的似然概率是所有条件概率的乘积。这意味着在整个序列上获得高似然概率需要每个条件概率都足够高。较长的序列更容易受到条件概率的累积效应的影响，而较短的序列可能更容易实现高似然概率，因为它们需要满足较少的条件。

2. **数据分布的影响**：在许多文本数据集中，较短的序列可能更常见，因为大多数文本片段通常比较短。这意味着模型在训练过程中更频繁地接触到较短的序列，从而更容易学习生成这些序列的条件概率。因此，较短的序列更有可能出现在生成过程中。

总之，较短的序列比较长的序列更有可能出现在通过序列的似然概率来度量文本模型质量的情况下，这是由于概率的乘积效应以及数据分布中较短序列的相对频繁性质所导致的。这并不一定表示模型质量差，而是反映了在模型训练和生成过程中的概率和数据分布的影响。

***
于是采用信息论：一个更好的语言模型应该能让我们更准确地预测下一个词元。
因此，它应该允许我们在压缩序列时花费更少的比特。
>交叉熵损失：交叉熵损失是一种常用于评估分类模型性能的指标，它衡量了模型的预测分布与真实分布之间的差异。在文本生成任务中，模型的预测分布表示模型在给定前面词元的情况下预测下一个词元的概率分布。

所以我们可以通过一个序列中所有的$n$个词元的交叉熵损失的平均值来衡量：

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
于是引出困惑度：

**困惑度**（Perplexity）是一种常用的度量模型质量的指标。它用来评估语言模型在生成文本时的性能，通常用于衡量模型生成文本的**准确度和流畅度**。

困惑度的定义如下：

1. 对于一个语言模型，给定一个文本序列$W$（通常表示为$W = w_1, w_2, \ldots, w_N$），困惑度度量了模型在生成该序列时的不确定性或困惑程度。
2. 困惑度是一个正数，它越小表示模型越自信且性能越好。较低的困惑度意味着模型更能够准确地预测下一个词元或字符。
3. 困惑度通常基于模型生成的概率分布来计算，其中模型预测下一个词元或字符的概率越高，困惑度越低。困惑度的计算公式通常如下：

   $$
   \text{Perplexity}(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_1, w_2, \ldots, w_{i-1})\right)
   $$

   其中，$P(w_i \mid w_1, w_2, \ldots, w_{i-1})$表示模型在给定前面词元的情况下预测第$i$个词元$w_i$的概率。

4. 困惑度可以理解为对数似然概率的几何平均值的倒数，因此它度量了模型生成文本序列的平均不确定性。



# RNN代码
## 从0实现
### RNN框架

用循环神经网络去创字符集语言模型。用困惑度来评估语言模型的质量。

```python
# 设置图表在Jupyter Notebook中直接显示
%matplotlib inline

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 1. 读取数据集
# 定义了一些常量，如批量大小(batch_size)和时间步数(num_steps)
batch_size, num_steps = 32, 35
# 加载了一段文本数据，得到了数据迭代器(train_iter)和词汇表(vocab)
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# 构建RNN网络
# 定义用于生成模型参数的函数
def get_params(vocab_size, num_hiddens, device):
    # vocab_size: 词汇表大小，用于确定输入和输出的维度
    # num_hiddens: 隐藏层的单元数，用于控制模型的复杂度
    # device: 指定计算设备，如CPU或GPU

    # 生成随机初始化的模型参数
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 生成标准正态分布随机数，用于初始化参数
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 将参数放入列表，并设置requires_grad为True，以便进行反向传播
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化RNN的隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    # batch_size: 批量大小，用于初始化隐藏状态的批量数
    # num_hiddens: 隐藏层的单元数，用于确定隐藏状态的维度
    # device: 指定计算设备，如CPU或GPU

    # 初始化RNN的隐藏状态，全零初始化
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义RNN的前向传播函数
def rnn(inputs, state, params):
    # inputs: 输入序列，每个时间步的输入
    # state: RNN的隐藏状态
    # params: RNN模型的参数

    # 获取参数
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 获取隐藏状态
    outputs = []
    for X in inputs:
        # RNN的前向传播
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 返回输出序列和最终的隐藏状态
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        """
        初始化函数，用于创建 RNNModelScratch 类的实例。

        Args:
            vocab_size: 词汇表的大小，表示模型可以生成的不同字符或词汇的数量。
            num_hiddens: 隐藏单元的数量，表示RNN中的隐藏状态的维度。
            device: 设备（CPU或GPU）用于运行模型。
            get_params: 生成模型参数的函数。
            init_state: 初始化 RNN 的隐藏状态的函数。
            forward_fn: RNN 的前向传播函数。

        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 调用 get_params 函数生成模型参数
        self.params = get_params(vocab_size, num_hiddens, device)
        
        # 保存初始化隐状态和前向传播函数
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        """
        模型的调用函数，用于进行前向传播。

        Args:
            X: 输入数据，通常是一个字符或词汇的索引序列。
            state: RNN 的当前隐状态。

        Returns:
            模型的输出和更新后的隐状态。
			模型的输出是一个包含了模型对下一个字符的预测概率分布的张量，其中每个元素对应于词汇表中一个字符的概率。为了生成一个字符，我们需要选择概率最高的字符，即选择概率分布中的最大值。
        """
        # 将输入 X 转化为 one-hot 编码并将数据类型转换为 float32
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        
        # 调用前向传播函数，返回输出和新的隐状态
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        """
        初始化 RNN 的隐藏状态。
        Args:
            batch_size: 批量大小，表示一次输入的样本数量。
            device: 设备（CPU或GPU）用于初始化隐状态。

        Returns:
            初始化后的隐状态。
        """
        return self.init_state(batch_size, self.num_hiddens, device)



```


1. **数据准备和预处理**：
   - 定义了批量大小和时间步数。
   - 使用`d2l`库加载时间机器的文本数据，并获取数据迭代器`train_iter`和词汇表`vocab`。

2. **模型参数初始化**：
   - 定义了函数`get_params`，用于生成随机初始化的循环神经网络(RNN)模型参数，包括权重矩阵和偏置。
   - 这些参数分为隐藏层参数（W_xh、W_hh、b_h）和输出层参数（W_hq、b_q）。

3. **初始化RNN隐藏状态**：
   - 定义了函数`init_rnn_state`，用于初始化RNN的隐藏状态，这里采用全零初始化。

4. **RNN前向传播**：
   - 定义了函数`rnn`，实现RNN的前向传播过程。
   - 该函数接收输入序列（`inputs`）、当前隐藏状态（`state`），以及模型参数（`params`）作为输入。
   - 在每个时间步中，通过更新隐藏状态和计算输出，将输入序列转化为输出序列。

5. **RNN模型类定义**：
   - 定义了一个名为`RNNModelScratch`的类，该类表示从零开始实现的RNN模型。
   - 构造函数`__init__`接收词汇表大小、隐藏层大小、设备、获取参数的函数、初始化隐藏状态的函数以及前向传播函数作为输入。
   - `__call__`方法用于执行模型的前向传播，将输入数据转化为输出数据。
   - `begin_state`方法用于初始化RNN的隐藏状态。

整个代码结构化地实现了一个RNN模型，它可以接受输入序列并将其转化为输出序列，同时提供了函数和类来初始化模型参数和隐藏状态。这种模型通常用于处理文本数据或其他序列数据。




对于独热编码：
简言之，是将每个索引映射为相互不同的单位向量：假设词表中不同词元的数目为$N$（即`len(vocab)`），词元索引的范围为$0$到$N-1$。如果词元的索引是整数$i$，那么我们将创建一个长度为$N$的全$0$向量，并将第$i$处的元素设置为$1$。此向量是原始词元的一个独热向量。
```python
F.one_hot(torch.tensor([0, 2]), len(vocab))  # 将输入的索引序列转化为独热编码，用于处理文本数据
```

我们每次采样的(**小批量数据形状是二维张量：（批量大小，时间步数）。**)`one_hot`函数将这样一个小批量数据转换成三维张量，（**时间步数，批量大小，词表大小**）的输出。这将使我们能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐状态。


### 梯度截断
对于长度为$T$的序列，我们在迭代中计算这$T$个时间步上的梯度，将会在反向传播过程中产生长度为$\mathcal{O}(T)$的矩阵乘法链。当$T$较大时，可能导致梯度爆炸或梯度消失。
因此，循环神经网络模型往往需要额外的方式来支持稳定训练。

***
>什么是利普希茨连续？

"利普希茨连续"（Lipschitz continuity）是一个在数学和函数分析中常用的概念，用于描述函数的连续性和导数的有界性。当一个函数被称为是"利普希茨连续"时，它满足以下性质：

给定两个点 x 和 y，存在一个正实数 L，使得对于这两个点，函数的值之间的差异不会超过 L 乘以这两个点之间的距离，即：

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

其中，L 被称为利普希茨常数，它表示了函数的“斜率”或“导数”的上界。这个定义表明，如果两个输入点 x 和 y 之间的距离很小，那么函数值之间的差异也会很小，差异受到常数 L 的限制。

利普希茨连续性是连续性的一种更强的形式。**如果一个函数是利普希茨连续的，那么它在一段距离内不会出现陡峭的波动，这对于数值优化、微分方程求解、机器学习等领域中的算法和数值方法非常重要。**
它保证了函数的变化不会突然变得很大，这有助于算法的稳定性和收敛性。


***


假设目标函数$f$在常数$L$下是*利普希茨连续的*（Lipschitz continuous）。
也就是说，对于任意$\mathbf{x}$和$\mathbf{y}$我们有：

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

在一次迭代中，我们将$\mathbf{x}$更新为$\mathbf{x} - \eta \mathbf{g}$，（$\mathbf{g}$为梯度，$\eta$为学习率）则：

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

这意味着我们不会观察到超过$L \eta \|\mathbf{g}\|$的变化。有两面性：
坏的方面，它限制了取得进展的速度；
好的方面，它限制了事情变糟的程度，尤其当我们朝着错误的方向前进时。

少数情况下，梯度$\mathbf{g}$过大，可以通过将梯度$\mathbf{g}$投影回给定半径（例如$\theta$）的球来裁剪梯度$\mathbf{g}$。
如下式：
**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**

好处：
1. 我们知道梯度范数永远不会超过$\theta$，并且更新后的梯度完全与$\mathbf{g}$的原始方向对齐。
2. 限制任何给定的小批量数据（以及其中任何给定的样本）对参数向量的影响，这赋予了模型一定程度的稳定性。
梯度截断是快速**修复梯度爆炸**的方法。(这不能阻止梯度消失噢)

```python
def grad_clipping(net, theta):
    """裁剪梯度
    Args:
        net: 神经网络模型或自定义的包含模型参数的对象。
        theta: 梯度裁剪的阈值。
    """
    # 检查输入的 net 是否是 nn.Module 类的实例
    if isinstance(net, nn.Module):
        # 获取网络中需要更新梯度的参数
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 如果 net 不是 nn.Module 的实例，假定它是一个自定义对象并获取其参数
        params = net.params

    # 计算所有参数梯度的 L2 范数（模）
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))

    # 如果梯度的 L2 范数大于阈值 theta，则进行梯度裁剪
    if norm > theta:
        for param in params:
            # 缩放参数的梯度以确保梯度的 L2 范数不超过阈值 theta
            param.grad[:] *= theta / norm

```


### 训练

1. 序列数据的不同采样方法（随机采样和顺序分区）将使用不同的隐状态初始化。
	1. 顺序分区，只在每轮**起始位置**初始化。因为他们是顺序分区的，隐状态可以直接顺延。并且由于都是顺延，会导致梯度计算复杂，所以在处理每个小批量数据之前，都要分离梯度。
	2. 随机抽样，每轮都要重新初始化。
2. 我们在更新模型参数之前裁剪梯度。这样的操作的目的是，即使训练过程中某个点上发生了梯度爆炸，也能保证模型不会发散。
3. 我们用困惑度来评价模型。


数据集：
![[Pasted image 20231105201040.png]]
通过将数据集print出来，发现就是上一个字母 对应 下一个字母。所以在我们的预测代码中一样，也是前一个output预测后一个output。



一轮训练过程：
```python
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    
    # 初始化状态 (state) 和计时器 (timer)
    state, timer = None, d2l.Timer()
    
    # 创建度量器，用于累积训练损失之和和词元数量
    metric = d2l.Accumulator(2)  # [训练损失之和, 词元数量]
    
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 如果使用nn.GRU，state是张量
                state.detach_()
            else:
                # 对于nn.LSTM或从零开始实现的模型，state是张量
                for s in state:
                    s.detach_()
        
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        y_hat, state = net(X, state)
        
        l = loss(y_hat, y.long()).mean()
        
        if isinstance(updater, torch.optim.Optimizer):
            # 如果使用内置的优化器，则进行参数梯度清零、反向传播和更新
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)  # 裁剪梯度，防止梯度爆炸
            updater.step()
        else:
            # 否则，进行反向传播和梯度更新
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        
        # 累积损失和词元数量
        metric.add(l * y.numel(), y.numel())
    
    # 计算困惑度 (perplexity) 和训练速度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

```

总训练过程：
```python
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（定义见第8章）"""
    
    # 交叉熵损失函数用于计算损失
    loss = nn.CrossEntropyLoss()
    
    # 创建用于动画的图表绘制工具
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    
    # 初始化模型参数优化器
    if isinstance(net, nn.Module):
        # 如果模型是 nn.Module 类的实例，使用 SGD 优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 否则，使用自定义的 sgd 函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    
    # 定义用于生成文本预测的函数
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 训练和预测过程
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            # 每10个迭代周期绘制一次动画图表
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    
    # 打印困惑度和速度信息
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```


预测代码
```python
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符

    Args:
        prefix: 一个字符串前缀，用于初始化模型的输入。
        num_preds: 生成的字符数量。
        net: 预训练的循环神经网络模型。
        vocab: 词汇表，用于将数字索引映射回字符。
        device: 设备（CPU或GPU），用于运行模型。

    Returns:
        生成的字符串，包括输入前缀和生成的字符。

    """
    
    # 初始化模型的隐状态
    state = net.begin_state(batch_size=1, device=device)
    
    # 初始化输出字符列表，并将前缀的第一个字符添加到列表中
    outputs = [vocab[prefix[0]]]
    
    # 定义一个函数用于获取模型输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 进行预热期，使用前缀初始化模型状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    
    # 开始生成新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1))
    
    # 将生成的字符索引映射回字符，并返回生成的字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])

```

它通过不断地输入模型的输出（字母索引） 作为下一个输入，以递归方式生成字符序列（采取前一个字母预测后一个字母的方式），同时保持模型的隐状态（state）

对于代码：
```python
outputs.append(int(y.argmax(dim=1).reshape(1))
```
模型的输出y是一个包含了模型对下一个字符的预测概率分布的张量，其中每个元素对应于词汇表中一个字符的概率。为了生成一个字符，我们需要选择概率最高的**字符**，即选择概率分布中的最大值。然后加入outputs，这时候加入的还是索引值。


训练
```python
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```


![[Pasted image 20231105191830.png]]

## 简洁实现

1. 读取数据集
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

2. 定义模型

构造一个具有**256个隐藏单元的单隐藏层**的循环神经网络层`rnn_layer`。
注意：*高级API的循环神经网络层返回一个输出和一个更新后的隐状态，我们还需要计算整个模型的输出层。

```python
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

***
稍加查看：


我们(**使用张量来初始化隐状态**)，它的形状是（隐藏层数，批量大小，隐藏单元数）。
```python
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```
torch.Size([1, 32, 256])

**通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。**
需要强调的是，这里`rnn_layer`的“输出”（`Y`）不涉及输出层的计算：它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。

```python
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

(torch.Size([35, 32, 256]), torch.Size([1, 32, 256]))

***
3. 定义完整的循环神经网络的类

注意，上文的`rnn_layer`只包含隐藏的循环层，我们还需要创建一个单独的输出层。

```python

class RNNModel(nn.Module):
    """循环神经网络模型"""
    
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        
        # 检查RNN是否是双向的（后文介绍），以确定全连接层的输入维度
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2 #如果是双向的
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):#它用于检查 self.rnn 是否不是 nn.LSTM 类的实例。
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device))

```

4. 训练

先用随机权重模型预测
```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

'time travellerktkpkpqpqt'
很明显不对。


真正训练过后：
```python
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

![[Pasted image 20231105192945.png]]


