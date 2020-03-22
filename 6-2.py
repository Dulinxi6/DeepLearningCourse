import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

boston_housing=tf.keras.datasets.boston_housing                     
(train_x,train_y),(_,_)=boston_housing.load_data(test_split=0)          #设置波士顿房价数据测试集所占比例为0

titles=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',              
        'RAD','TAX','PTRATIO','B','LSTAT']
plt.figure(figsize=(12,16))                                             #创建12*16的画布

for i in np.arange(13):                                                 
    plt.subplot(4,4,i+1)                                                #将画布划分为4*4个子图
    plt.scatter(train_x[:,i],train_y)
    plt.title(str(i+1)+'.'+titles[i]+'-Price')
    plt.xlabel(titles[i])
    plt.ylabel("Price(-$1000's)")
    plt.ylim(0,60)
plt.suptitle('各个属性与房价的关系',fontsize=14)
plt.rcParams["font.sans-serif"]="SimHei"
plt.tight_layout(rect=[0,0,1,0.9])
plt.show()

for j in np.arange(13):
    print(str(j+1)+'--'+titles[j])
inp=int(input('请选择属性：'))
for m in np.arange(1,14):
    if inp==m:
        plt.scatter(train_x[:,m-1],train_y)
        plt.title(str(m)+'.'+titles[m-1]+'-Price')
        plt.xlabel(titles[m-1])
        plt.ylabel("Price(-$1000's)")
        plt.ylim(0,60)
plt.show()

