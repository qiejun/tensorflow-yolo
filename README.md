# tensorflow-yolo
yolo v1 with tensorflow 
### 实现过程：<br>
输入图像尺寸为[448,448],训练网络为自己搭建的，fine-tuning vgg16，只调用了vgg16的卷积层，之后自己又添加了卷积层和全连接层，为了加速收敛，引入了BN层。损失函数按照YOLO论文实现，训练数据为voc2007，训练共迭代了100个epoch。
### 测试结果：<br>
<img src="https://github.com/qiejun/tensorflow-yolo/blob/master/pictures/b.jpg" width="200" height="200" alt="img"><img src="https://github.com/qiejun/tensorflow-yolo/blob/master/pictures/a.jpg" width="200" height="200" alt="img"><img src="https://github.com/qiejun/tensorflow-yolo/blob/master/pictures/c.jpg" width="200" height="200" alt="img">
