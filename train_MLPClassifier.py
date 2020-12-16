# %%
#读数据画图
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X=np.loadtxt(r"1x0.txt")
y=np.loadtxt(r"1y0.txt")

from pylab import *
plot(X[y==1,0],X[y==1,1],'b^')
plot(X[y!=1,0],X[y!=1,1],'ro')
show()

# %%
#数据分组，预处理，-1~1之间

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.1)
aver=Xtrain.mean(axis=0)
Xtrain-=aver
Xtemp=np.abs(Xtrain)
colmax=Xtemp.max(axis=0)
Xtrain/=colmax

Xtest-=aver
Xtest/=colmax

#对分类信息正交化，输出层两个节点
labelstrain = []
for i in range(len(ytrain)):
    one=2*[0]
    if ytrain[i]==0:
        one[0]=1  #  1  0
    else:
        one[1]=1  #  0  1
    labelstrain.append(one)
labelstrain=np.array(labelstrain)
# %%
clf=MLPClassifier(hidden_layer_sizes=(100,), random_state=7,verbose=1)
clf.fit(Xtrain, labelstrain)#建模
yhat=clf.predict(Xtest)
print(np.argmax(yhat,axis=1)) # 预测值
print(ytest)

# %%
