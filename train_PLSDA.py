# %%
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn import datasets
iris=datasets.load_iris() # 从数据库获得数据
data=iris.data #获得自变量数据
target=iris.target  # 获得样本的分类信息
# 选择两类鸢尾花出来
x=data[target!=2]
y=target[target!=2]
y[y==0]=-1

# 样本分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2)

av=X_train.mean(axis=0)
std=X_train.std(axis=0)
X_train=(X_train-av)/std

pls = PLSRegression(n_components=3, scale=False) # 数据已经标准化
pls.fit(X_train, y_train)
T=pls.x_scores_

X_test = (X_test-av)/std
Yhat=pls.predict(X_test)[:,0]
Yhat[Yhat<0]=-1
Yhat[Yhat>0]=1
err=y_test-Yhat
errRate=len(err[err!=0])/len(err)*100
print(errRate)

import matplotlib.pyplot as plt
f1=0
f2=1

Tpred=None  # 测试集的得分
for i in range(2):
    t=X_test @ pls.x_weights_[:,i]
    if Tpred is None:
        Tpred=t
    else:
        Tpred=np.c_[Tpred,t]
    X_test=X_test-np.outer(t,pls.x_loadings_[:,i])
# 画测试集合散点图
plt.scatter(Tpred[:, 0], Tpred[:, 1], c=y_test,  edgecolors='black', s=25)
# 训练集画图
plt.scatter(T[y_train==1,f1],T[y_train==1,f2],c='b',marker='o',label='good')
plt.scatter(T[y_train==-1,f1],T[y_train==-1,f2],c='r',marker='v',label='bad')
plt.legend(loc='lower right')
plt.show()

# %%
