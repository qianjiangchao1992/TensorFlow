# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:31:36 2018

@author: Administrator
"""
#以线性回归为例，构建Tensorflow计算图
import tensorflow as tf
#第一步，给出X，Y
x_train=tf.constant([1.,2.,3.])
y_train=tf.constant([1.,2.,3.])
#第二步设置变量，w,b
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='basis')
#给出假设函数
hypothesis=w*x_train+b
#cost=1/m*∑(hypothesis-y_train)^2
#cost loss function
cost=tf.reduce_mean(tf.square(hypothesis-y_train))
#采用梯度下降法
optimize=tf.train.GradientDescentOptimizer(learning_rate=0.01) #采用tf.train中的梯度下降法
train=optimize.minimize(cost)#建立训练模型
sess=tf.Session()  #开启会话
sess.run(tf.global_variables_initializer()) #初始化所有变量
for step in range(2001):
    sess.run(train)   #开始训练模型
    if step%20==0:
        print(step,sess.run(cost),sess.run(w),sess.run(b))
write=tf.summary.FileWriter('regession_01_logs',sess.graph)
write.close()

