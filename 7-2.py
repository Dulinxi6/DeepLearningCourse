import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist                       
(train_x,train_y),(test_x,test_y)=mnist.load_data()

plt.figure(figsize=(16,16))
#创建画布
for i in np.arange(16):
    ran=np.random.randint(0,len(test_x))
    plt.subplot(4,4,i+1)
    plt.axis('off')
    plt.title('标签值:' + str(i+1))
    plt.imshow(test_x[ran])

plt.suptitle('MNIST测试集样本',color='red',fontsize=20)
plt.rcParams['font.sans-serif']='SimHei'
plt.show()
