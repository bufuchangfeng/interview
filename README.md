# interview

## 机器学习基础

* 偏差和方差

* 评价指标

  auc、pr曲线、roc

* bagging、boosting、stacking

* batch normalization和layer normalization

* L1正则和L2正则、L0正则

* 聚类

* 过拟合、欠拟合

* KL散度、VC维

* 先验概率、条件概率、贝叶斯公式

* 生成模型和判别模型

* 最小二乘法

* SVD

* 损失函数

* svr（svm用于回归）

* gridsearch

* cross validation

* 共轭分布

* 岭回归和Lasso

* 各种模型对缺失值的处理

* 交叉熵和相对熵

  




## 深度学习基础

* early stopping

* dropout

* optimizer

  weight decay

  momentum

* 模型压缩

* loss震荡

* skip-connection

* 神经网络反向传播推导（非矩阵形式）

* 1*1卷积作用

* 权值共享

* 卷积求参数量

* 卷积核求输出维度

* 模型蒸馏

* AutoML

* 膨胀卷积

* focal loss

* 反卷积

* 转置卷积

* im2col

* cuda

* 空洞卷积

* 深度学习中的norm

* cnn的参数共享

* 感受野

  怎么计算感受野？怎么增加感受野？

  


## NLP

* word2vec、glove、fasttext

* textcnn

* eda数据增强

  

## 算法

* HMM

* CRF

* PCA

* LDA（线性判别分析）

* kmeans

* knn

* AutoEncoder

* attention

* transformer

* bert、ELMO、GPT

* 朴素贝叶斯

* dbscan

*  svm

  带核函数的svm求解得到的模型其实是支持向量，预测时将待预测样本与支持向量计算距离。

* tfidf
* bm25
* kmeans++
* resnet
* vgg
* alexnet
* googlenet
* lenet
* densenet
* mobilenet
* lda
* 概率图
* GMM（高斯混合模型）
* Factor Machine（因子分解机）
* textrank
* pagerank
* plsa
* em
* one-class svm
* GMM-HMM

  





## 算法与数据结构

* 堆
* 红黑树
* 堆排序
* 逆波兰表达式求值
* 中缀变后缀
* topK问题
* 桶排序
* 逆序数
* 一个链表，奇数位升序，偶数位降序
* 给一个01二项分布的随机器，参数为p，用它设计一个0-1的均匀分布的随机器（连续的）
* Trie
* avl树
* 求第k大数
* 线段树
* y = sqrt(x)

  





## 真实问题

* LSTM为什么能解决梯度消失和梯度爆炸

* 机器学习中有哪些算法需要进行归一化

* softmax减去最大数字不变的证明

  实际场景做softmax很容易出现下溢问题，这个可以用每个维度减去一个固定值就可以了

* python深拷贝和浅拷贝

* python中的反射

* tensorflow和pytorch的区别

* 为什么用拟牛顿法代替牛顿法

  牛顿法和拟牛顿法有哪些区别

* pooling层的反向传播

* 卷积层的反向传播

* python中is和==的区别

* 神经网络初始化方法

* cnn并行

* python的垃圾回收机制

* java的垃圾回收机制

* 机器学习偏差和方差是什么，从欠拟合到过拟合二者如何变化

* xgboost为啥用二阶泰勒展开

* linux系统怎么查看进程cpu使用率

* pytorch里dataset、dataloader、sampler有什么区别

* 逻辑回归为啥要做特征离散化

* 讲一下梯度和导数的区别

* 什么是线性模型？LR为什么是线性模型？

* 如何判断机器是大端模式还是小端模式

* SVM如何做文本分类

* lr不做标准化有影响吗，神经网络呢？

* 梯度下降法于牛顿法的优缺点？

* 随机森林的随机怎么理解

* MSE和交叉熵的区别

* 写线性回归的解析解，矩阵不可逆怎么办

* 如何处理数据中的缺失值？这些处理方法有什么区别

* lr为什么不用min square loss

* 使用hinge loss的意义，为什么linear svm的bound要设为1？

* 什么是kernel trick？对应无限维空间可以使用哪种kernel function？

* LR 并行化怎么做？

* LightGBM和XGBoost是怎么处理缺失值的？

* 解决样本不平衡的方法

* LR加上正则化项后怎么求解

* LR和 SVM哪个对异常点比较敏感

  SVM和LR对于离群点的敏感性

* 模型融合的时候，如果每个分类器正确率为0.5，投票法能不能得到正确率0.95

* c++值传递和引用传递区别

* 常用分类算法中哪些是线性的哪些是非线性的

* 集成学习的基分类器从决策树换成svm或者lr可不可以，为什么

* relu存在死亡神经元，如何解决

* 梯度下降和牛顿法在具体问题中如何选择

* svm如何进行多分类，多分类hinge loss什么形式

* 介绍LR、RF、GBDT ，分析它们的优缺点，是否写过它们的分布式代码

* LR如何引入非线性

* L1正则不可导

* 特征出现共线性怎么办？你是直接把他们删掉吗？怎么发现共线性？

* 权重衰减等价于哪个正则项

  


## 编程语言

* python的底层实现
* c++
* java
* python魔法函数
* python装饰器
* java中.equals()和==的区别
* c++智能指针
* HashMap的实现原理
* python 为什么list不能作为dict的key
* python的gil
* python生成器、迭代器、装饰器
* python lambda
* python列表和元组



## 操作系统

* 进程和线程的区别
*  用户态和内核态
* lru算法
* 进程间通信
* linux awk  tail grep

  

## 计算机网络

* TCP三次握手

  

## 智力题

* 三个瓶子倒水问题。11升，5升，6升的瓶子，其中11升的瓶子里装满了水，请倒出8升的水。
* 有一个rand可以等概率产生1—7这7个数字，如何利用这个rand等概率的产生1—9
* 有一个0-1的均匀分布随机器，用它实现一个N(0, 1)的正态分布随机器
* 9枚硬币，8枚一样重，1枚比较重，最少称几次能找到最重的那枚？2次

  

## 经典论文

* 《Neural Machine Translation By Jointly Learning To Align and Translate》
* 《Deep Neural Networks for YouTube Recommendations》

  

## 概念

* 自监督学习
* 弱监督学习
* 无监督学习
* gan
* 强化学习

  

## 需要自学的课程

* 设计模式
* 编译原理
* 图神经网络
