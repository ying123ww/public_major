# 前言
![[Pasted image 20231116150824.png]]

总目标，提高模型的泛化能力。
**训练误差**是指模型在训练数据上的性能，而**泛化误差**是指模型在未见过的数据上的性能。**过拟合**是一种现象，指模型在训练数据上表现优越，但在未见过的数据上表现较差，通常是因为模型过于复杂，学到了训练数据的噪声和细节。
为了评估模型的泛化能力，数据通常被划分为**训练集**（用于模型训练）、**验证集**（用于调整模型参数和防止过拟合）、**测试集**（用于评估泛化性能）。模型在训练集上学习，在验证集上进行调整和选择，在测试集上进行最终评估。
**正则化**是一种降低过拟合风险、提高泛化能力的方法，通过在模型的损失函数中引入额外的惩罚项，限制模型的复杂性，以提高其在新数据上的表现。验证集在这个过程中发挥关键作用，帮助选择合适的模型和参数，以平衡训练误差和泛化误差，防止过拟合。
# Introduction
## 定义
介绍挺清楚的网页：
https://www.e-works.net.cn/report/ml/ml.html

定义：
机器学习（Machine Learning，简称ML）是一种人工智能（AI）的分支，它关注计算机系统通过学习经验数据来改善性能。机器学习的主要目标是通过模式识别和学习算法，使计算机系统能够自动进行学习和适应，而无需明确地进行编程。


步骤：
1. **数据收集：** 收集与特定任务相关的数据，这些数据包含了系统需要学习的信息。
2. **特征提取：** 选择或设计适当的**特征**，这些特征用于描述数据中的关键信息，以便算法能够进行学习。
3. **模型训练：** 使用训练数据来训练机器学习模型，模型通过学习数据中的模式和规律来进行调整。
4. **模型评估：** 使用测试数据评估模型的性能，检查模型在未见过的数据上的泛化能力。
5. **预测或决策：** 使用训练好的模型对新数据进行预测或做出决策。


机器学习算法分三类：

1. **监督学习（Supervised Learning）：** 在监督学习中，算法接受**带有标签**的训练数据，学习输入与输出之间的映射关系。目标是让算法能够对新的、未标记的数据进行准确的**预测**。
2. **无监督学习（Unsupervised Learning）：** 无监督学习中，算法使用**没有标签**的数据，试图发现数据中的模式和结构。**聚类和降维**是无监督学习的两个主要任务。
3. **强化学习（Reinforcement Learning）：** 强化学习是一种学习方式，其中算法通过与环境进行交互，根据反馈信号来调整其行为。目标是使算法在环境中获得**最大的累积奖励**。

机器学习按照建模方式分类：

1. **参数化模型：**
   - **定义：** 参数化模型假定模型的结构是事先定义好的，但模型的参数是可以从训练数据中学习得到的。
   - **特点：** 这类模型有一个**固定数量的参数**，不随着训练数据的增加而增加。通常，参数化模型对于较小规模的数据集较为适用。
   - **例子：** 线性回归、逻辑回归、线性支持向量机等。
2. **非参数化模型：**
   - **定义：** 非参数化模型的**结构通常不是事先固定**的，而是会根据训练数据的复杂性而变化。这类模型的参数数量可以根据数据量的增加而增加。
   - **特点：** 非参数化模型更加灵活，能够适应**更复杂的数据模式，但在处理较小规模的数据时可能过拟合**。
   - **例子：** k近邻、决策树、随机森林、核密度估计等。


选取依据：

- **数据规模：** 对于较小的数据集，参数化模型可能更适用，因为其参数数量有限，不容易过拟合。
- **数据复杂性：** 如果数据的真实模式比较复杂，非参数化模型可能更能够捕捉到这种复杂性。
- **计算资源：** 非参数化模型通常需要更多的计算资源，因为它们的灵活性可能导致模型更加复杂。




机器学习过程示例：
![[Pasted image 20231116135548.png]]


## 区别ML、DL、AI
![[Pasted image 20231116134423.png]]

- 人工智能（Artificial Intelligence，简称AI）是一个更广泛的概念，涵盖了机器学习和深度学习等多个领域。人工智能旨在使计算机系统表现出类似人类智能的能力，包括学习、推理、问题解决、语言理解、感知和自主行动等方面。
- 我们主要区别ML和DL
	1. **定义：**
	   - **机器学习：** 是一种通过对数据进行学习和模式识别来改进系统性能的技术。它涵盖了多种算法和方法，包括监督学习、无监督学习、半监督学习和强化学习等。
	   - **深度学习：** 是机器学习的一种特殊形式，它使用深度神经网络来学习和提取数据的特征。深度学习侧重于使用多层神经网络（深度神经网络）来**模拟和解决复杂的问题**。
	
	2. **特征表示：**
	   - **机器学习：** 通常需要**手工选择或设计特征**，以便算法能够从数据中学到有用的信息。
	   - **深度学习：** 通过多层神经网络**自动学习和提取特征**，不需要手动设计特征。
	
	3. **数据需求：**
	   - **机器学习：** 在相对**较小的数据集**上表现良好，有时不需要大量的数据。
	   - **深度学习：** 通常需要**大量的标记数据**来训练深度神经网络，以获得良好的性能。
	
	4. **计算需求：**
	   - **机器学习：** 在相对简单的硬件和计算资源上运行。
	   - **深度学习：** 对于大型神经网络和复杂模型，通常需要更多的计算资源，如图形处理单元（**GPU**）。
	
	5. **适用范围：**
	   - **机器学习：** 适用于各种任务，包括**分类、回归、聚类和推荐**等。
	   - **深度学习：** 在处理大规模数据和复杂模式识别问题时表现较好，如图像识别、语音识别、自然语言处理等。


# 基本思想

为了评估模型的泛化性能，通常将数据集划分为训练集、验证集和测试集。模型在训练集上进行训练，利用验证集进行调优，最后在测试集上评估模型的泛化误差。

控制过拟合的一种常见方法是使用正则化技术，它通过在模型的损失函数中引入额外的惩罚项来限制模型的复杂性。这有助于防止模型过度拟合训练数据，提高其泛化能力。


## 训练误差和泛化误差
**泛化能力**是让模型在学习过程中不仅要关注训练数据，而且要能够有效处理新、未见过的数据的能力。对于一个模型来说，我们肯定希望它泛化能力较好。

训练误差（Training Error）和泛化误差（Generalization Error）是机器学习中两个重要的概念，它们有助于评估模型在训练和未见过的数据上的性能。

1. **训练误差：**
   - **定义：** 训练误差是**模型在训练数据集上的表现**，即模型对用于训练的数据的拟合程度有多好。
   - **计算：** 训练误差通常通过计算模型在训练数据上的损失或错误来衡量，损失越小或准确率越高，训练误差就越小。

2. **泛化误差：**
   - **定义：** 泛化误差是模型在**未见过的数据上的表现**，即模型对新数据的预测能力有多好。
   - **计算：** 泛化误差通常通过计算模型在验证集或测试集上的损失或错误来估计，这些数据与模型训练过程中使用的数据是不同的。

在理想情况下，希望模型能够在训练数据上取得低的训练误差，同时也在未见过的数据上取得低的泛化误差。然而，过度拟合（Overfitting）是一个常见的问题，即模型在训练数据上表现良好，但在未见过的数据上表现较差。过度拟合的原因可能是**模型过于复杂**，过多地学习了训练数据的噪声和细节，而没有很好地泛化到新数据。
![[Pasted image 20231116150824.png]]


## 过拟合

1. **模型复杂性**: 模型的复杂性是指模型的能力以捕捉数据中的模式。一个高度复杂的模型可以捕捉数据中的细微差别和复杂模式。

2. **过拟合的概念**: 过拟合发生在模型不仅学习了数据中的真实模式，还学习了数据中的噪声和随机波动。这意味着模型在训练数据上表现得非常好，但是对于新的、未见过的数据表现较差。

3. **复杂性与过拟合的关系**: 当模型过于复杂时，它有更多的参数可以调整以适应训练数据。这可能导致模型学习到数据中的“噪声”，而不仅仅是真实的、普遍存在的模式。因此，虽然模型在训练集上的表现可能很好，但这主要是因为它“记住”了数据的特定特征，而不是学习到了可以泛化到新数据上的通用规律。

4. **平衡模型复杂性**: 为了避免过拟合，需要在模型的复杂性和它在未见数据上的表现之间找到平衡。这通常通过正则化技术（如L1或L2正则化）、选择合适的模型复杂度或使用交叉验证来实现。

5. **示例与直觉**: 想象一个试图通过多项式函数拟合一组数据点的情况。一个低阶多项式（如线性或二次）可能无法捕捉所有的模式（欠拟合），而一个高阶多项式可能会在数据点之间做出极端的波动，从而精确匹配每个点（过拟合）。最佳的模型通常在这两个极端之间，能够捕捉足够的模式，同时保持对新数据的泛化能力。
![[Pasted image 20231116235506.png]]
所以总之：
**过拟合**指模型**过度地拟合到了观测数据中噪声的部分**，以至于在未观测的数据标签预测时出现较大偏差的现象。

## 训练集、测试集、验证集

数据通常被分为三个主要部分：训练集、验证集和测试集。

1. **训练集 (Training Set)**：
   - **用途**：训练集用于训练模型，即通过这些数据来学习和构建模型的参数。
   - **重要性**：训练集越大且多样化，模型学习的机会就越多。但过大的训练集可能导致**过拟合**。

2. **验证集 (Validation Set)**：
   - **用途**：验证集用于模型调优和选择。在模型训练过程中，验证集用来**检验模型的性能，比如调整超参数（如学习率、网络层数等）**。
   - **重要性**：验证集提供了一种评估模型在训练过程中的表现的方式，而不会影响最终测试集上的表现。这**有助于选择最好的模型和避免过拟合**。

3. **测试集 (Test Set)**：
   - **用途**：测试集用于**最终评估模型的性能**。这些数据在整个训练和验证过程中都是不可见的。
   - **重要性**：测试集提供了一种客观的模型性能衡量方法。

**平衡三者的关系**：
- **数据分割**：在实践中，数据通常按照某种比例（如70%训练、15%验证、15%测试）划分。
- **迭代过程**：模型在训练集上学习，在验证集上进行调整和选择，在测试集上进行最终评估。

```python
from sklearn.model_selection import train_test_split
import numpy as np

# 假设 X 是您的特征数据，y 是您的标签
# 例如：X, y 可能是通过 pandas 读取的 DataFrame 数据
# X = df.drop('label_column', axis=1)
# y = df['label_column']

# 首先将数据分为训练+验证集和独立的测试集
	X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 然后将训练+验证集进一步划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=42)

# 打印各数据集的大小以验证
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

```
首先将数据集划分为85%的训练+验证集和15%的测试集，然后将85%的训练+验证集进一步划分为82%的训练集和18%的验证集.

对于`train_test_split` 的解释：
1. 第一和第二个参数 (通常是 X 和 y):
   - 这两个参数通常是 X（特征集）和 y（目标变量或标签）。
   - X 包含了数据的特征，通常是一个二维数组（或类似的结构），如 `DataFrame` 或 `ndarray`。
   - y 包含了每个数据点的目标或标签，通常是一个一维数组。

2. **`test_size`**:
   - 这个参数指定了测试集占总数据集的比例。
   - 它可以是一个浮点数，表示测试集大小的比例，例如 0.2 意味着 20% 的数据将作为测试集。
   - 也可以是一个整数，直接指定测试集中的样本数量。

3. **`train_size`**:
   - 类似于 `test_size`，但它指定的是训练集占总数据集的比例。
   - 可以是一个比例（浮点数）或直接是样本数量（整数）。
   - 通常，只设置 `test_size` 或 `train_size` 其中一个就足够了，因为两者会自动互补（例如，如果 `test_size=0.25`，则 `train_size` 自动为 0.75）。

4. **`random_state`**:
   - 这个参数是一个整数或 RandomState 实例，用于控制随机数生成器的种子。
   - 设置 `random_state` 可以确保每次分割的结果都是一致的，这对于实验的可重复性非常重要。

5. **`shuffle`**:
   - 这个参数决定是否在分割之前随机打乱数据。
   - 默认为 True，即数据会被随机打乱，然后分割。
   - 如果设置为 False，则按原始顺序分割数据（这在某些时间序列数据分析中可能是必要的）。

6. **`stratify`**:
   - 这个参数通常用于分类任务中，以确保训练集和测试集具有相似的类别比例。
   - 例如，如果设置为 y，则数据分割时会尽量保持 y 中每个类别的比例。
   - 这对于处理不平衡数据集非常有用。

## 正则化

正则化是机器学习中用来防止过拟合的一种技术，它通过添加额外的信息（通常是一种惩罚项）来约束或限制模型。以下是一些常用的正则化技术：

1. **L1 正则化（Lasso 正则化）**:
   - 在 L1 正则化中，正则化项是模型所有权重的绝对值之和。
   - 它可以导致模型中某些权重变为零，从而实现特征的选择。
   - L1 正则化对于创建稀疏模型（即大多数特征权重为零）很有用。

2. **L2 正则化（Ridge 正则化）——权重衰减**:
   - L2 正则化的正则化项是所有权重的平方和。
   - 这种方法不会使权重变为零，但会推动权重向小的值移动，从而减少模型复杂度。
   - L2 正则化对于控制过拟合非常有效，特别是当模型非常复杂时。


5. **Dropout**:
   - Dropout 是深度学习中常用的一种正则化技术，尤其在训练神经网络时。
   - 在每个训练步骤中，随机选定的神经元被“丢弃”（即它们在该步骤中的激活被设为零）。
   - 这样可以防止模型对特定的节点过度依赖，增强模型的泛化能力。

6. **早停（Early Stopping）**:
   - 早停并不是正则化技术，但它是一种用于避免过拟合的方法。
   - 在训练过程中，如果验证集的性能不再提升，或者开始降低，则提前终止训练。
   - 这样可以保证模型不会过度学习训练数据。

7. **批归一化（Batch Normalization）**:
   - 虽然主要用于加速深度网络的训练，批归一化也可以间接作为正则化来使用。
   - 它通过规范化层的输入来减少内部协变量偏移，这有助于稳定和加速神经网络的训练。

### L1正则化&L2正则化
![[Pasted image 20231117002138.png]]
这张图展示了在机器学习中使用 L1 正则化（Lasso）和 L2 正则化（Ridge）时损失函数的变化。每个子图都表示了参数空间中的损失函数，其中 θ0 和 θ1 分别是模型的两个参数。

- **左图**：只显示了原始损失函数 $J_0(\theta)$的等高线，没有正则化项。这里的 $\theta^*$表示损失函数的最小值点，即模型参数的最优解。

- **中图**：展示了加入 L2 正则化项后的损失函数 $J_0(\theta) + \lambda_2||\theta||^2$的等高线。L2 正则化倾向于选择更小的参数值，因此最优解 $\theta^*$移向了原点。等高线更加圆润，意味着参数值的变化不会引起损失函数剧烈变化，从而减少模型的复杂度。

- **右图**：展示了加入 L1 正则化项后的损失函数 $J_0(\theta) + \lambda_1||\theta||_1$的等高线。L1 正则化导致等高线在参数轴上形成了角点，这意味着最优解倾向于在这些轴上，因此它会产生稀疏的解，即某些参数值会变成零。

在这三个图中，黑点 $\theta^*$表示没有正则化时损失函数的最小值点，蓝色虚线表示加入正则化项后的损失函数等高线。红色虚线在中图和右图中表示正则化项的等高线，分别对应 L2 和 L1 正则化。你会注意到，当正则化项加入后，损失函数的最小值点（即最优参数值）发生了改变。这反映了正则化如何影响模型参数的估计。

总结一下，图中展示了通过加入不同的正则化项，模型参数的最优值是如何被约束向零移动的。L1 正则化因其产生稀疏解而被用于特征选择，而 L2 正则化因其能够处理模型参数的大幅波动而被广泛使用。




权重衰减是一种常用的正则化技术，属于 L2 正则化的范畴。在权重衰减中，正则化项是模型权重的平方和，这个项被添加到模型的损失函数中。权重衰减的主要目的是减少模型权重的大小，以防止过拟合。

在数学上，权重衰减通过在损失函数中添加一个与权重平方成比例的项来实现。这个额外的项会惩罚较大的权重值。在优化过程中，这导致了权重的缩减或“衰减”，从而帮助减少模型的复杂度。

具体来说，如果原始损失函数是 $L $，权重向量是 $\mathbf{w}$，权重衰减系数是 $\lambda$，那么加入权重衰减后的损失函数 $L'$可以表示为：
$$L' = L + \frac{\lambda}{2} \|\mathbf{w}\|^2 $$

这里，$\|\mathbf{w}\|^2$是权重向量的 L2 范数（即权重的平方和），$\frac{\lambda}{2}$是控制正则化强度的系数。权重衰减系数 $\lambda $的值越大，对模型复杂度的惩罚就越大。

在实际应用中，特别是在训练神经网络时，权重衰减是一个非常常见且有效的方法来防止模型过拟合。通过适当选择 $\lambda$的值，可以在模型的复杂度和泛化能力之间找到一个平衡点。


