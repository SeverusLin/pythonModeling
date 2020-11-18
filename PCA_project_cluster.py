# %%
import matplotlib.pyplot as plt
import numpy as np
X=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_X.txt')
av=X.mean(axis=0)
std=X.std(axis=0)
X=(X-av)/std
y=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_Y.txt')
# %%
from PCA import PCA
pca=PCA(X)
print(pca.SVDdecompose()) # 返回比值
T,P=pca.PCAdecompose(3)
f1=0
f2=1
plt.scatter(T[y==1,f1],T[y==1,f2],c='b',marker='o',label='good')
plt.scatter(T[y==-1,f1],T[y==-1,f2],c='r',marker='v',label='bad')
plt.legend(loc='upper left')
plt.show()

# %%
