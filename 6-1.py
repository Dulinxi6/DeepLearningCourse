import matplotlib.pyplot as plt

area = [137.7,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21]
price= [145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30]

plt.rcParams["font.sans-serif"]="KaiTi"                     #设置字体为楷体          
plt.rcParams["axes.unicode_minus"]=False                    #正常显示字符

plt.scatter(area,price,marker='o',color='red')              #创建散点图，数据点为圆点，颜色为红色
plt.title('商品房销售记录',color='blue',fontsize=16)
plt.xlabel('面积（平方米）',fontsize=14)
plt.ylabel('价格（万元）',fontsize=14)
plt.xlim(40,150)
plt.ylim(30,140)

plt.show()