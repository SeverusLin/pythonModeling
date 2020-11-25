# %%
# 数据预处理
import numpy as np
from sklearn import datasets
iris=datasets.load_iris() 
data=iris.data
target=iris.target 
X=data[target!=2]
y=target[target!=2]
print(y)

# 正交编码
ally=[]
for i in range(len(y)):
    if (y[i]==1):
        oneY=[0,1]# 第二类
    else:
        oneY=[1,0] # y[i]==0
    ally.append(oneY)
Y=np.array(ally)

# 主成分回归
from PCR import PCR

pcr=PCR(X,Y)
print(pcr.confirmPCs())
pcr.fit(3)
yPred=pcr.predict(X)  # 预报样本的函数值，每个样本2个值
exp = np.exp(yPred)
sumExp = exp.sum(axis=1, keepdims=True)  # 保持二维矩阵格式
softmax = exp / sumExp   #  可以输出看每类的确认概率
yPred=np.argmax(softmax,axis=1)
print(yPred)
# %%
# 计算误报率
err=y-yPred
err=err[err!=0]
errRate=len(err)/len(y)
print(errRate)
# 生成格点
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
T_sim=np.c_[t0,t1]  # 作为T矩阵
X_sim=T_sim @ P[:,0:2].T   #  X=T @ P.T
Yhat=pcr.predict(X_sim)
exp=np.exp(Yhat)
sumExp=exp.sum(axis=1,keepdims=True)
softmax = exp / sumExp
Z = softmax [:, 0]  # 第一类的概率
# %%
# 制作概率的等高线图
import matplotlib.pyplot as plt
Z = Z.reshape(xx.shape)
CS = plt.contour(xx,yy, Z, 10, colors='k',)             
cls1=y==1
cls2=y==0
plt.plot(T[cls1,0],T[cls1,1],'ro')
plt.plot(T[cls2,0],T[cls2,1],'b^')

plt.clabel(CS, fontsize=9, inline=1)
plt.show()


