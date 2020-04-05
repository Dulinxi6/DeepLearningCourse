import numpy as np
import tensorflow as tf

x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

x=tf.constant(x)            #创建张量
y=tf.convert_to_tensor(y)   #创建张量的第二种方法
fz=fm=0
sumx=sumy=0
for i in np.arange(0,len(x)):
    sumx+=x[i]
    sumy+=y[i]
avgx=sumx/len(x)
avgy=sumy/len(y)
for i in np.arange(0,len(x)):
    fz+=(x[i]-avgx)*(y[i]-avgy)
    fm+=(x[i]-avgx)**2
W=fz/fm
b=avgy-W*avgx
print(np.array(W))
print(np.array(b))