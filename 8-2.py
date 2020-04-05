import tensorflow as tf
import numpy as np

x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

x=tf.constant(x)            #将数组x,y转换为张量
y=tf.convert_to_tensor(y)
sumf1=sumf2=sumf3=summ1=summ2=0

for i in np.arange(0,len(x)):
    sumf1+=x[i]*y[i]
    sumf2+=x[i]
    sumf3+=y[i]
    summ1+=x[i]**2
    summ2+=x[i]
W=(len(x)*sumf1-sumf2*sumf3)/(len(x)*summ1-summ2**2)
b=(sumf3-W*sumf2)/len(x)
print(np.array(W))
print(np.array(b))
