# %%[markdown]
# # 利用REG库实现Cobb-Douglas函数的回归

# %%
import numpy as np
from REG import MLR 
# %%
data=np.loadtxt(r"C:\Users\51770\Desktop\pythonModeling\data3.txt")
# 选取数据文件路径，根据数据文件位置修改
X=data[:,:2]
Y=data[:,2:]

X_log=np.log(X)
Y_log=np.log(Y)
# %% [markdown]
# ## 模型拟合

# %%

mlr=MLR(X_log,Y_log)
mlr.fit()

print(np.e**mlr.a) # Cobb-Douglas 函数的几个参数
# %% [markdown]
# ## 模型预测
# %%
Yhat_log=mlr.predict(X_log)
# %% [markdown]
# ## 绝对误差
# %%
err=abs(np.e**Yhat_log-np.e**Y_log)/np.e**Y_log*100
print(err)
# %% [markdown]
# ## F-检验
# %%
print(mlr.FTest(0.01))
# %% [markdown]
# ## 协方差
# %%
print(mlr.COV(0.01))