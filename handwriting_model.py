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
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(200,4 ), alpha=1e-5, random_state=1,max_iter=40000)#最大迭代次数
clf.fit(X, Y)  # 训练模型
import pickle
with open('handWriting.bin', 'wb') as f:
    rs = pickle.dumps(clf)
    f.write(rs)

# %%
from PIL import Image
import pickle
import numpy as np
from ImageDigit import ImageDigit
with open('handWriting.bin', 'rb') as f:
    clf=pickle.load(f)

# %%
#  预测一个手写图片
img = Image.open(r"3.jpg")
imgToDigit=ImageDigit(img)
imgToDigit.convert_to_bw(200)
digits=imgToDigit.split()
imgToDigit.to_32_32()
Xnew,Ynew=imgToDigit.featureExtract_new()
predictions=clf.predict(Xnew)  # 每个样本的预报类别标识
print('预测值：',np.argmax( predictions,axis=1))

# %%
# 灰度直方图
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
im=Image.open(r"2.jpg").convert('L')
# L 为256级灰度模式，1 二值模式，
im.show()
img=np.array(im)
plt.figure("hist")
arr=img.flatten()
n, bins, patches = plt.hist(arr, bins=256, density =1, facecolor='green', alpha=0.75)  
plt.show()

# %%
from PIL import Image
def convert_to_bw(im,threshold):  #im是图像对象
	im = im.convert("L")  # 转换为256级灰度图像
	im = im.point(lambda x: 255 if x > threshold else 0) #大于阈值，取白色，否则像素值置0，黑色
	im = im.convert('1') # 黑白二值图像
	return im
im = Image.open(r"2.jpg")
im=convert_to_bw(im,143)
im.show()

# %%
#画图像分割线
from PIL import Image
import numpy as np
im=Image.open(r"2.jpg").convert('L')
imOrig = im.point(lambda x: 255 if x > 143 else 0) 
im = imOrig.convert('1')
imData=np.array(im)
im_arr = 1-imData  # 黑点1，其他0
xmax, ymax = im.size
row_arr = im_arr.sum(axis=1) # 行有多少个黑点
col_arr = im_arr.sum(axis=0)  # 每列有多少个黑点
row_arr = row_arr[1:]  # 第一行去掉
row_boundary_list = []  # 找行分割线
for i in range(len(row_arr)-1):  # 最后一行也去掉
    if row_arr[i] == 0 and row_arr[i+1] != 0 or row_arr[i] != 0 and row_arr[i+1] == 0:
        row_boundary_list.append(i+1)

column_boundary_list = []  # 找列分割线
for i,x in enumerate(col_arr[:-1]):
    if (col_arr[i] == 0 and col_arr[i+1] != 0) or col_arr[i] != 0 and col_arr[i+1] == 0:
        column_boundary_list.append(i+1)

imData=np.array(imOrig)
imData[row_boundary_list]=1  # 显示分割线
imData[:,column_boundary_list]=1
img = Image.fromarray(imData.astype('uint8')).convert('L')
img.show()

# %%
