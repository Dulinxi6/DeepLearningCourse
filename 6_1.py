import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

#导入数据
df=pd.read_csv("data/boston.csv",header=0)
ds=df.values

x_data=ds[:,:12]
y_data=ds[:,12]
#对特征值做（0-1）归一化
for i in range(12):
    x_data[:,i]=(x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())
#划分数据集：训练集、验证集和测试集
train_num=300  
valid_num=100  
test_num=len(x_data)-train_num-valid_num  
#训练集划分
x_train=x_data[:train_num]
y_train=y_data[:train_num]

#验证集划分
x_valid=x_data[train_num:train_num+valid_num]
y_valid=y_data[train_num:train_num+valid_num]

#测试集划分
x_test=x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test=y_data[train_num+valid_num:train_num+valid_num+test_num]
#数据转换
x_train=tf.cast(x_train,dtype=tf.float32)
x_valid=tf.cast(x_valid,dtype=tf.float32)
x_test=tf.cast(x_test,dtype=tf.float32)

#构建模型
#定义模型
def model(x,w,b):
    return tf.matmul(x,w)+b
W=tf.Variable(tf.random.normal([12,1],mean=0.0,stddev=1.0,dtype=tf.float32))
B=tf.Variable(tf.zeros(1),dtype=tf.float32)

#训练模型
#设置训练参数
training_epochs=50  
learning_rate=0.001 
batch_size=10       

#定义均方差损失函数
def loss(x,y,w,b):
    err=model(x,w,b)-y  #计算模型预测值和标签值的差异
    squared_err=tf.square(err)  
    return tf.reduce_mean(squared_err) 
#计算样本数据【x,y】在参数【w,b】点上的梯度
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,w,b)
    return tape.gradient(loss_,[w,b]) 
#创建优化器
optimizer=tf.keras.optimizers.SGD(learning_rate)

#迭代训练
loss_list_train=[] 
loss_list_valid=[] 
total_step=int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs=x_train[step*batch_size:(step+1)*batch_size,:]
        ys=y_train[step*batch_size:(step+1)*batch_size]
        
        grads=grad(xs,ys,W,B) #计算梯度
        optimizer.apply_gradients(zip(grads,[W,B])) #优化器根据梯度自动调整变量w和b
    loss_train=loss(x_train,y_train,W,B).numpy()
    loss_valid=loss(x_valid,y_valid,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d},train_loss={:.4f},valid_loss={:.4f}".format(epoch+1,loss_train,loss_valid))
plt.figure(figsize=(6,6))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'blue',label="Train Loss")
plt.plot(loss_list_valid,'red',label="Valid Loss")
plt.legend(loc=1)
plt.show()