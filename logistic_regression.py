# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:33:12 2018

@author: qjcaho
"""
#逻辑回归
import tensorflow as tf
#第一步给出X,Y值
x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]
#placeholders for a tensor that will be always fed
x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])
#给出变量w,b
w=tf.Variable(tf.zeros([2,1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='basis')
#给出目标函数H(X)
hypothesis=tf.sigmoid(tf.matmul(x,w)+b)
#给出损失函数cost
cost=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
#调用梯度下降法去最优解损失函数cost
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#Accuracy computation
#True if hypothesis>0.5 else false
predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))
#launch graph
with tf.Session() as sess:
    #初始化所有的tensor变量
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        cost_val,_=sess.run([cost,train],feed_dict={x:x_data,y:y_data})
        if step%200==0:
            print(step,cost_val)
    #Accuracy report
    h,c,a=sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data,y:y_data})
    print("\nhypothesis: ",h,"\npredicted: ",c,"\naccuracy: ",a)
write=tf.summary.FileWriter('logistic_regression_logs',sess.graph)
write.close()
