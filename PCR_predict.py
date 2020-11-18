# %%
import matplotlib.pyplot as plt
import numpy as np
X=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_X.txt')
av=X.mean(axis=0)
std=X.std(axis=0)
X=(X-av)/std
Y=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_Y.txt')
allY=[]
for i in Y:
    if i==1.0:
        allY.append([1,0])
    else:
        allY.append([0,1])
y=np.array(allY)
# %%
from PCR import PCR

pcr=PCR(X,y)
print(pcr.confirmPCs())
pcr.fit(8)
yPred=pcr.predict(X)  # 预报样本的函数值，每个样本2个值
exp = np.exp(yPred)
sumExp = exp.sum(axis=1, keepdims=True)  # 保持二维矩阵格式
softmax = exp / sumExp   #  可以输出看每类的确认概率
yPred=np.argmax(softmax,axis=1)
print(yPred)

# %%
Y[Y==1.0]=0
Y[Y==-1.0]=1
err=Y-yPred
err=err[err!=0]
errRate=len(err)/len(y)
print(errRate)

# %%
T=pcr.T
P=pcr.P
x_min = T[:, 0].min() - .5
x_max = T[:, 0].max() + .5
y_min = T[:, 1].min() - .5
y_max = T[:, 1].max() + .5
h = .2  
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
t0=xx.flatten()  # 平铺
t1=yy.flatten()  # 平铺
Tmoni=np.c_[t0,t1]  # 作为T矩阵
Xmoni=Tmoni @ P[:,0:2].T   #  T @ P.T   得到X
Yhat=pcr.predict(Xmoni)
exp=np.exp(Yhat)
sumExp=exp.sum(axis=1,keepdims=True)
#sumExp = np.sum(exp, axis=1, keepdims=True)
softmax = exp / sumExp
Z = softmax [:, 0]  #选择第一类的概率输出
# 制作概率的等高线图
Z = Z.reshape(xx.shape)
CS = plt.contour(xx,yy, Z, 10, colors='k',) # 负值将用虚线显示             
cls1=Y==0
cls2=Y==1
plt.plot(T[cls1,0],T[cls1,1],'ro')
plt.plot(T[cls2,0],T[cls2,1],'b^')

plt.clabel(CS, fontsize=9, inline=1)
plt.show()



# %%
