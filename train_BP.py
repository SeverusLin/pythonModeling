# %%
#读数据画图
import numpy as np
from sklearn.model_selection import train_test_split
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
#建模
from NeuralNetwork import NeuralNetwork
nn =NeuralNetwork( [X.shape[1],100,labelstrain.shape[1] ],
'logistic')
nn.fit(Xtrain,labelstrain,epochs=10000)

# 参数[X.shape[1],100,labelstrain.shape[1]]，用输入特征数，和输出特征数，定义网络拓扑。

#预测及分类报告
predictions=nn.predict(Xtest)  # 每个样本的预报类别标识
print(predictions)
print(ytest)
from sklearn.metrics import classification_report
print (classification_report(ytest, predictions))

# %%
