import numpy as np
x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
sum_x=sum_y=0         
for i in np.arange(len(x)):
    sum_x=sum_x+x[i]
    sum_y=sum_y+y[i]
sum_x=sum_x/len(x)
sum_y=sum_y/len(y)
fz=fm=0
for i in np.arange(len(x)):
    fz+=(x[i]-sum_x)*(y[i]-sum_y)  
    fm+=(x[i]-sum_x)**2         
W=fz/fm
b=sum_y-W*sum_x
print('W值为：%.2f,b的值为：%.2f' %(W,b))