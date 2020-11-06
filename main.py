# %%
from REG import MLR
import numpy as np 

A=np.loadtxt(r"C:\Users\51770\Desktop\pythonModeling\data.txt")

x=A[:,0:-1] # 取一列不要写成A[:,0]
y=A[:,1:]
mlr=MLR(x,y)
mlr.fit()

yHat=mlr.predict(x)
print(mlr.a)

# %%
err=abs(yHat-y)/y*100
print(err)

# %%
print(mlr.FTest(0.01))

# %%
print(mlr.RTest(0.01))

# %%
print(mlr.COV(0.01))

