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

# 2019.3.7

## tensorflow

- linear regression:
  
  ```python
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())
  ```

  在tensorflow定义中，流动的输入输出数据用placeholder定义，系统中需要更新的参数用Variable。预设时可以不设定具体的tensor尺寸，包括参数定义时，可以不定义numpy的随机变量生成的尺寸，可见tensorflow对numpy有很好的支持。

  