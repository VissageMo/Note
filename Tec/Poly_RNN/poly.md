# Polygen RNN
由加拿大多伦多大学 David Acuna 等人提出，使用 LSTM 网络，解决数据集自动标注和人机交互标注的问题。算法主要包括三部分：CNN 编码器，LSTM 解码器， GGNN 优化。
<div align=center>

![](imgs/20181129-092310.png)
</div>

##主要结构
### CNN编码器
该部分采用Resnet-50的结构，为了在提取高阶信息的同时保留低阶的边界信息，将不同