import numpy as np
class NeuralNetwork:
 # 定义激活函数，生长函数及其一阶导数
    def logistic(self,x):
        return 1/(1 + np.exp(-x))
    def logistic_derivative(self,x):
        return self.logistic(x)*(1-self.logistic(x))
    #定义双曲函数及其一阶导数
    def tanh(self,x):
        return np.tanh(x)
    def tanh_deriv(self,x):
        return 1.0 - np.tanh(x)**2
    # 构造函数
    # layer用一个元组传递，输入、  隐、出 节点数
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = self.logistic
            self.activation_deriv = self.logistic_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_deriv = self.tanh_deriv
    #两个层之间，有权重，初始化
        self.weights = []
        for i in range(len(layers) - 1):
        #np.random.random((n,m)),产生n*m矩阵，每个数是0-1随机小数
            # *2后，成为0-2之间，再-1，变成-1到+1之间。
            if i==0: # 输入层，加偏置，节点数+1
                w=2*np.random.random((layers[0]+1 , layers[1]))-1
                self.weights.append(w*0.5)
            else:
                w=2*np.random.random((layers[i] , layers[i + 1]))-1
                self.weights.append(w*0.5) 
    #X矩阵，每行是一个样本 ，y是分类，#learning_rate 学习速率， epochs最大迭代次数
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        temp = np.ones(X.shape[0]) #初始化矩阵
        X = np.c_[X,temp]   #加1列1，为bias
        for k in range(epochs):
            #随机选取一个样本，对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [   X[i]  ]  # 变成列表
            #完成所有正向的输出的计算
            for j in range(len(self.weights)):
                a.append(self.activation(a[j] @ self.weights[j] ))  # 巧妙
            error = y[i] - a[-1]  # 输出层在a[-1]
            deltas = [error * self.activation_deriv(a[-1])]  # 输出层调节初值，见下式红框
            #开始反向误差传播，更新权重
            for j in range(len(a) - 2, 0, -1): # 从倒数第2层到开始层
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):  # 更新权重
                layer = np.atleast_2d(a[i])    #  作为推导公式的第③ ，见前面推导
                delta = np.atleast_2d(deltas[i])  # 向量转成只有一行的矩阵
                self.weights[i] += learning_rate * layer.T @ delta 
    #预测函数
    def predict(self, x):  # 给定一组预测样本  x
        temp = np.ones(x.shape[0])
        x=np.c_[x,temp]  #加偏置
        ans=[]
        for a in x:
            for w in self.weights:
                a = self.activation(np.dot(a, w))
            ans.append(  np.argmax(a)  )  # 输出层节点正交化转类
        return ans
