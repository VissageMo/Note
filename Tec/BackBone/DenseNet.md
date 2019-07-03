# DenseNet

## 概述

作为CVPR2017的Best Paper，DenseNet在ResNet的基础上更进一步，不止在当前层和上一层建立连接，而是将所有层的结果进行concat，通过特征重用和旁路（Bypass）设置，既大幅度减少网络参数，又一定程度上缓解了gradient vanishing的问题。
