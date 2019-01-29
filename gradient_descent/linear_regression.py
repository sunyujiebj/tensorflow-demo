# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据
points_num = 100
vectors = []

# 用Numpy 的正太随机分布函数生成100个点
# 这些点的（x,y）坐标值对应线性方程 y = 0.1 * x + 0.2
# 权重（Weight） 0.1,偏差 「Bias」0.2

for i in xrange(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors] # 真实点的 X 坐标
y_data = [c[1] for c in vectors] # 真实点的 Y 坐标

# 图像1： 展示100个随机数据点
plt.plot(x_data, y_data, 'r*', label= "Orifinal data")
plt.title("Linear")
plt.legend()
plt.show()

# 构建线性回归模型

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 初始化 Weight
b = tf.Variable(tf.zeros([1]))  # 初始化 Bias
y = W * x_data + b  #模型计算出来的y

# 定义 loss functuion (损失函数) 或 cost function （代价函数）
# 对Tensor 的所有维度 都去计算((y - y_data）^ 2) 之和 / N
loss = tf.reduce_mean(tf.square(y - y_data))

#使用梯度下降的优化器来优化 loss function ()
optimizer = tf.train.GradientDescentOptimizer(0.5) # 设置学习率0.5
train = optimizer.minimize(loss)

# 创建会话
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练一定的步数
for step in xrange(20):
    #优化每一步
    sess.run(loss)
    #打印每一步的损失权重 偏差
    print("step=%d, Loss=%f, [Weight=%f Bias=%f]") \
        % (step, sess.run(loss), sess.run(W), sess.run(b))

# 图像 2 ： 绘制所有的点并且绘制出最佳拟合的直线
plt.plot(x_data, y_data, 'r*', label= "Orifinal data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted Line")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#关闭会话
sess.close()