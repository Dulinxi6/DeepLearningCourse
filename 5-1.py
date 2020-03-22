import numpy as np
import numpy.matlib as np
np.random.seed(612)                 #设置种子                        
ran=np.random.uniform(0,1,1000)     #产生1-1000成正太分布的随机数组
num=int(input('输入1-100之间的整数：'))
print('序号\t索引值\t\t随机数')
m=0                                 #m为序号
for i in np.arange(1000):
    if i%num==0:
        m+=1
        print('%d\t%d\t%.16f' %(m,i,ran[i]))