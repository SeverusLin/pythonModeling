# %%
import numpy as np
from ImageDigit import ImageDigit
from PIL import Image

img = Image.open("2.jpg")
imgToDigit=ImageDigit(img)
imgToDigit.histShow()
thr=int(input('请输入背景阈值：'))
imgToDigit.convert_to_bw(thr)
digits=imgToDigit.split()
imgToDigit.to_32_32()
X,Y=imgToDigit.featureExtract_new()
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
ma=X.max(0)  # 求每列的最大值
ma[ma==0]=1  # 有很多列是0，保证不被0除而出错
mi=X.min(0);X=X-mi ;X=X/ma
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1) 
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100, ), alpha=1e-5, random_state=1,max_iter=20000)#最大迭代次数
clf.fit(X_train, y_train)  # 训练模型
score = clf.score(X_test, y_test)
print('预测得分：',score)
yhat = clf.predict(X_test)
print("预测值",np.argmax(yhat,axis=1))
print("真  值",np.argmax(y_test,axis=1))

# %%
