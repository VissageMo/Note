# 面试准备计划

## 复习内容

- 代码能力
  - tensorflow、keras、pytorch
  - pandas、numpy 数据处理
  - python 基本语法
  - leetcode 算法题

- 知识储备
  - 机器学习基础
    - 多层感知机 ：小蓝书只有感知机推导，多层感知机见花书、西瓜书
    - K 临近搜索
    - 支持向量机
    - 神经网络
    - 随机森林
    - 隐马尔科夫
    - 条件随机场
  - 图像算法
    - 深度学习框架
    - 传统图像处理
  - 经典书籍
    - 花书
    - 小蓝书 ：完成 1-2 章
    - 剑指 offer

- 实习内容
  - 深圳实习：deeplab v3的使用
  - 机械臂算法：AlexNet 的使用

## Daily Report

### 2019.3.7

#### tensorflow

- linear regression:
  
    ```python
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())
    ```

  在tensorflow定义中，流动的输入输出数据用placeholder定义，系统中需要更新的参数用Variable。预设时可以不设定具体的tensor尺寸，包括参数定义时，可以不定义numpy的随机变量生成的尺寸，可见tensorflow对numpy有很好的支持。

### 2019.4.22

- **torchsummary**

    实用的pytorch网络结构显示插件

    ``` python
    model = resnet()
    torchsummary.summary(model, (1, 256, 256))
    ```

- **resnet**

    记得计算输出的尺寸，保证最后一层每个特征图尺寸为1

### 2019.4.26

- **1x1 conv**

    1x1卷积核具有三个作用：打通特征图之间的关系、改变特征图层数（减少数据量）、解决输入图像需要固定尺寸的问题

    1. 降低训练参数

        在ResNet的BottNeck结构中，通过256x1x1x64的卷积层，将特征图减少到64层，后进行64x3x3x64的卷积，之后再通过64x1x1x256回到256层，有效减少参数量

    2. 实现跨通道的交互与信息融合

        传统的卷积层中每个滤波器需要同时映射跨通道和空间维度的相关性，包含2个空间维度信息（图像的长和宽）和一个通道维度信息。因此研究者考虑可以将传统的卷积层操作分离开来，分别从空间维度和通道维度提取特征。

        ![img](imgs/1.jpg)

        Extreme Inception module首先使用一个1x1卷积核映射跨通道相关性，然后分别映射每一个输出通道的空间相关性，而Xception module则使用一种深度可分卷积运算，首先提取每个通道上的空间相关性（depthwise convolution），然后再经过1x1卷积核获取跨通道的相关性（pointwise convolution）。

        ![img](imgs/2.jpg)
        ![img](imgs/3.jpg)

- **GAP**

    全局平均池化（Global Average Pooling）用于代替FC层输出分类，在保持精确性几乎不变的前提下压缩参数量

    思路：在最后一层生成n个卷积层，n为输出分类数量。对每个卷积层参数取平均，作为结果进行softmax后输出。

    缺陷：可能导致收敛速度变慢，在反向传播梯度时，前层的每个点梯度将变为1/m^2

### 2019.4.27

- **知识蒸馏 knowledge distillation**

    [blog](https://blog.csdn.net/nature553863/article/details/80568658)

    Hinton的文章"Distilling the Knowledge in a Neural Network"首次提出了知识蒸馏（暗知识提取）的概念，通过引入与教师网络（teacher network：复杂、但推理性能优越）相关的软目标（soft-target）作为total loss的一部分，以诱导学生网络（student network：精简、低复杂度）的训练，实现知识迁移（knowledge transfer）。

    通过提出的 soft target 辅助 hard target 进行训练，因为 hard target 虽然绝对正确，但包含信息量很低，而 soft target 包含信息量大。（比如同时分类驴和马的时候，尽管某张图片是马，但是soft target就不会像hard target 那样只有马的index处的值为1，其余为0，而是在驴的部分也会有概率）

    这样的好处是，这个图像可能更像驴，而不会去像汽车或者狗之类的，而这样的soft信息存在于概率中，以及label之间的高低相似性都存在于soft target中。但是如果soft targe是像这样的信息[0.98 0.01 0.01]，就意义不大了，所以需要在softmax中增加温度参数T（这个设置在最终训练完之后的推理中是不需要的）

    $$ q_i = \frac{exp(z_i/T)}{\sum\limits_{j}\exp(z_j/T)} $$
    $$ L = \alpha L^{(soft)}+(1-\alpha)L^{(hard)} $$

    ![img](imgs/6.png)

### 2019.4.29

- **Pytorch Styleguide**
  
  [GithubLink](https://github.com/IgorSusmelj/pytorch-styleguide)

  1. 常用程序库

        prefetch_generator: Loading next batch in background during computation

        tqdm: Progress during training of each epoch

  2. 文件组织

        不要将所有的层和模型放在同一个文件中。最好的做法是将最终的网络分离到独立的文件（networks.py）中，并将层、损失函数以及各种操作保存在各自的文件中（layers.py，losses.py，ops.py）。最终得到的模型（由一个或多个网络组成）应该用该模型的名称命名（例如，yolov3.py，DCGAN.py），且引用各个模块。

- **Smooth L1**

    用于图像检测问题的 Loss 结构，最早在 Fast RCNN 中提出，负责对回归框的位置误差进行评价。

    $$ \lambda\frac{1}{N_{reg}}\sum\limits_ip_i^*L_{reg}(t_i,\:t_i^*) $$

  - $i$ ： mini-batch 的 anchor 的索引。
  - $pi$：目标的预测概率。
  - $pi^∗$：target二分类是否有物体，有物体为1，否则为0。
  - $ti$：一个四点向量，预测坐标
  - $ti^∗$：是一个四点向量，是ground truth boungding box的坐标（真实坐标）
    $$ smooth_{L_1} = \left\{\begin{aligned}
    &0.5x^2, \qquad if|x|<1\\
    &|x|-0.5,\qquad otherwise\\
    \end{aligned}\right.$$

### 2019.4.30

- **Yolo V3**

    Yolo 检测速度快，但对于小物体效果比较差。

### 2019.5.3

- **pytorch loss function** [link](https://blog.csdn.net/zhangxb35/article/details/72464152?utm_source=itdadao&utm_medium=referral)

    对pytorch而言，很多 Loss 都有 size_average 和 reduce 两个参数。reduce为False，直接返回向量形式的Loss；reduce为True，size_average为True，返回平均自，否则返回之和。

    ```python
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
    loss_fn = torch.nn.SmoothL1Loss()
    loss_fn = torch.nn.MSELoss()
    ```

  - nn.L1loss
  
    $$ loss(x_i, y_i) = |x_i-y_i| $$

  - nn.SmoothL1Loss

    $$ loss(x_i,y_i)=\left\{\begin{aligned}
        &0.5(x_i-y_i)^2 \qquad if|x_i-y_i| < 1\\
        &|x_i-y_i|-0.5 \qquad otherwise\\
    \end{aligned}\right.$$

  - nn.MSELoss

    $$ loss(x_i,y_i)=(x_i-y_i)^2 $$

  - nn.BCELoss

    二分类的交叉熵函数，使用时需要在输入前加sigmoid，

    $$ loss(x_i,y_i)=-w_i[y_ilogx_i+(1-y_i)log(1-x_i)] $$

  - nn.

  - 

- **激活函数**
  
    主要包括：sigmoid，softmax，relu

  - relu

    优点：求导简单，配合SGD下降速度快，梯度稳定，解决梯度爆炸和梯度消失问题

    缺点：导数不连续，可能出现 dead relu 现象

    - dead relu

        dead relu，指当输入小于零时，输出为零，反向传播时没有梯度，节点无法更新，保持为零。

        解决方法：
          1. 降低学习率（最本质）
          2. 使用L2正则化，或其他优化器（Adam）
          3. 优化的relu，如：Leaky ReLU

    - Leaky ReLU

        $$ ReLU = max(0, x)\\
           Leaky \ ReLU = max(\lambda x, x) \qquad \lambda < 1 $$

        由于在小于0的位置，也会有梯度，可以较好地解决 dead relu 问题