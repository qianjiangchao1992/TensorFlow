# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:23:14 2018

@author: qjc

title:线性回归代码
"""
import tensorflow as tf
#设计线性模型
#假设函数H(x)=wx+b
#step1:X_train AND Y_train
X_train=[1,2,3]
Y_train=[1,2,3]
#step2:设置W and b，设置变量（最后能知道W,b) 而占位符当计算结束后数据不在
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
#step3:设置假设函数
#H(x)=W*X_train+b
hypothesis =W*X_train+b
#step4:设置损失函数
#cost=∑(H(x)-y)^2/m
cost=tf.reduce_mean(tf.square(hypothesis-Y_train))
#step5:采取优化方式最优化损失函数cost
#Minize GradientDescent
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
#step6:开启会话
sess=tf.Session()
########重要######
####初始化变量#####
sess.run(tf.global_variables_initializer())
# Fit the line
for step in range(4001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))
sess.close()
write=tf.summary.FileWriter(r'F:\gitdatabase\TensorFlow\regression01\logistic_regression_logs',sess.graph)
write.close()

###实现图看命令
#1.打开Anaconda prompt 输入：activate tensorflow
#2.接着输入:tensorboard  --logdir=路径（C:\Users\Administrator\Desktop\logistic_regression_logs）  --host=127.0.0.1
#3 然后复制命令行中http://127.0.0.1:6006




