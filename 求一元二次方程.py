#求一元二次方程:ax^2+bx+c=0
import math
a=int(input('输入a的值:'))
b=int(input('输入b的值:'))
c=int(input('输入c的值:'))
print('输入的a,b,c的值分别为：{} {} {}'.format(a,b,c))
k=b**2-4*a*c
if a==0:
    print('请重新输入a的值:')
    a=int(input('请输入a:'))
if k>0:
    x1=(-b+math.sqrt(k))/(2*a)
    x2=(-b-math.sqrt(k))/(2*a)
    print("方程有两个根x1,x2的值分别为:{:.2f}、{:.2f}".format(x1,x2))
if k==0:
    x=(-b)/(2*a)
    print('方程有一个根x的值为：{:.2f}'.format(x))
if k<0: 
    print('方程无解')
