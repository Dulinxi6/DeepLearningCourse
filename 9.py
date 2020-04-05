#第9讲 单元作业
import tensorflow as tf
import numpy as np

x1 = np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x2 = np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2],dtype=np.float32)
y = np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
x0=np.ones(len(x1))
X=tf.convert_to_tensor(np.stack((x0,x1,x2),axis=1))
#讲x0,x1,x2进行堆叠
Xt=tf.transpose(tf.convert_to_tensor(X))
#求X的转置
Y=tf.convert_to_tensor(y.reshape(-1,1))

W=tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(Xt,X)),Xt),Y)
W=np.array(W).reshape(-1)

a=float(input('商品房面积（20-500）：'))
rn=int(input('房间数（1-10）：'))
t1=t2=1
while t1 or t2: 
    if a<20 or a>500:
        print('请重新输入面积：')
        a=float(input('商品房面积（20-500）：'))
    else :
        t1=0
    if rn<1 or rn>10:
        print('请重新输入房间数：')
        rn=int(input('房间数（1-10）：'))
    else :
        t2=0
y_pred=W[1]*a+W[2]*rn+W[0]
print('预售价为：',y_pred)