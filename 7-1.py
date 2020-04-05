import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image
img=Image.open('lena.png')
img=img.resize((256,256))
img_r,img_g,img_b=img.split()

#img_r.thumnial
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
img_rn=img_r.resize((50,50))
plt.imshow(img_rn,cmap='gray')
plt.axis('off')
plt.title('R-缩放')

plt.subplot(2,2,2)
img_gn=img_g.transpose(Image.FLIP_LEFT_RIGHT)
img_ga=img_gn.transpose(Image.ROTATE_270)
plt.imshow(img_ga,cmap='gray')
plt.title('G-镜像+旋转',fontsize=14)

plt.subplot(2,2,3)
img_bn=img_b.crop((0,0,150,150))
plt.axis('off')
plt.imshow(img_bn,cmap='gray')
plt.title('B-裁剪',fontsize=14)

plt.subplot(2,2,4)
img=Image.merge('RGB',[img_r,img_g,img_b])
plt.axis('off')
plt.imshow(img)
plt.title('RGB',fontsize=14)

plt.rcParams['font.sans-serif']='SimHei'
plt.suptitle('图像基本操作',color='blue',fontsize=20)
plt.show()