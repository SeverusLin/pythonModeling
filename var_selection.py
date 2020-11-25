# %%
from sklearn.feature_selection import SelectPercentile,f_classif
import numpy as np

X1=X=np.loadtxt(r"C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_X.txt")
y=np.loadtxt(r"C:\Users\51770\Desktop\pythonModeling\wheat_train_PCA_Y.txt")
avg=X.mean(axis=0);
std=X.std(axis=0);X=(X-avg)/std
selector2= SelectPercentile(f_classif, 50)  # 选择50%的变量
Xnew=selector2.fit_transform(X, y)  # 用选择的变量重构矩阵

print(selector2.pvalues_)
print(selector2.get_support())
indx=np.argwhere(selector2.get_support())[:,0]
print(indx)

# %%
from PCA import PCA
pca=PCA(Xnew)  
ans=pca.SVDdecompose()
print(ans[1])
pca.plotScore(y,yAxis=2,inOne=True)

# %%
