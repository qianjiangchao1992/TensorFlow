import tensorflow as tf
import numpy as np
import pandas as pd
import time 
#1.加载数据集把对输入和结果分开
train=pd.read_csv(r'E:\tensorflowtraindata\train.csv')
image=train[train.columns[1:]].values
labels_flat=train[train.columns[0]].values.ravel()
#2.对输入进行处理
images=image.astype(np.float)
images=np.multiply(images,1.0/255.0)
print('输入的数量为(%g,%g)'%(images.shape))
image_size=images.shape[1]
print('输入数据的维度=> {0}'.format(image_size))
image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('图片的长=> {0}\n图片的高=> {1}'.format(image_width,image_height))
x=tf.placeholder('float',shape=[None,image_size])

#3.对结果进行处理
labels_count=np.unique(labels_flat).shape[0]
print('结果的种类=> {0}'.format(labels_count))
y=tf.placeholder('float', shape=[None, labels_count])
#进行one——hot编码
def dense_one_hot(labels_dense,num_classes):
    labels_num=labels_dense.shape[0]
    index_offset=np.arange(labels_num)*num_classes
    labels_one_hot=np.zeros((labels_num,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot
labels=dense_one_hot(labels_flat,labels_count)
labels=labels.astype(np.uint8)
print('结果的数据集为：({0[0]},{0[1]})'.format(labels.shape))
#把输入数据划分为验证集和训练集
vatidation_size=2000
validate_image=image[:vatidation_size]
validate_labels=labels[:vatidation_size]
train_image=image[vatidation_size:]
train_labels=labels[vatidation_size:]
#5对训练集进行分批
batch_size=100
n_batch=len(image)/batch_size
print(n_batch)
#6创建一个简单神经网络对图片进行识别
weights=tf.Variable(tf.zeros([784,10]))
basis=tf.Variable(tf.zeros([10]))
result=tf.matmul(x,weights)+basis
print(result)
prediction=tf.nn.softmax(result)
#7创建损失函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#8用梯度下降法优化参数
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#9初始化变量
init=tf.global_variables_initializer()
#计算准确度
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    #一定要初始化
    sess.run(init)
    #循环50轮
    for epoch in range(20):
        stat_time=time.time()
        for batch in range(int(n_batch)):
            #按照分片取数
            batch_x=train_image[batch*batch_size:(batch+1)*batch_size]
            batch_y=train_labels[batch*batch_size:(batch+1)*batch_size]
            #进行训练
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        end_start=time.time()
        accuracy_n=sess.run(accuracy,feed_dict={x:validate_image,y:validate_labels})
        print('第'+str(epoch+1)+'轮，准确度=>'+str(accuracy_n))
        print('第'+str(epoch+1)+'轮，时间=>'+str(end_start-stat_time))
write=tf.summary.FileWriter('logs', sess.graph)
write.close()

#打开终端命令--输入


(base) C:\Users\Administrator>activate tensorflow
(tensorflow) C:\Users\Administrator>tensorboard --logdir=D:\data\tensorflow\tens
orflow\MNIST_data\logs --host=127.0.0.1
d:\programdata\anaconda3\envs\tensorflow\lib\importlib\_bootstrap.py:222: Runtim
eWarning: numpy.dtype size changed, may indicate binary incompatibility. Expecte
d 96, got 88
  return f(*args, **kwds)
TensorBoard 1.9.0 at http://127.0.0.1:6006 (Press CTRL+C to quit)
