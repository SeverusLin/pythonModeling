# %%
from sklearn.cross_decomposition import PLSRegression
import numpy as np
X=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_X.txt')
av=X.mean(axis=0)
std=X.std(axis=0)
X=(X-av)/std
Y=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_Y.txt')

# 利用PCA算出主成分累计百分比确定主成分数
pls = PLSRegression(n_components=6, scale=False) # 数据已经标准化
pls.fit(X, Y)
T=pls.x_scores_

import matplotlib.pyplot as plt
f1=0
f2=2
plt.scatter(T[Y==1,f1],T[Y==1,f2],c='b',marker='o',label='good')
plt.scatter(T[Y==-1,f1],T[Y==-1,f2],c='r',marker='v',label='bad')
plt.legend(loc='upper left')
plt.show()

# %%
