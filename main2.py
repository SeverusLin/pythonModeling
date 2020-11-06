# %%
import numpy as np
from REG import MLR

data=np.loadtxt(r'C:\Users\51770\Desktop\pythonModeling\data2.txt')

x=data[:,0:4]
y=data[:,4:]
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
B=np.linalg.svd(x,full_matrices=True)
print(B[1])

# %%
print(mlr.COV(0.01))
