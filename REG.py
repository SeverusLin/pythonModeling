#线性回归
import numpy as np
from scipy.stats import f
class MLR:
    def __init__(self,X,Y,intercept=True): # 构造函数 
        self.X = X
        self.Y = Y
        self.n_x = X.shape[0]
        self.k_x = X.shape[1]
        self.n_y = Y.shape[0]
        self.k_y = Y.shape[1]
        self.intercept = intercept
    def fit(self): # 线性回归
        if self.intercept:
            ones=np.ones(self.n_x)
            A=np.c_[ones,self.X]
        else:
            A=self.X
        self.a=np.linalg.inv(A.T@A)@(A.T@self.Y)
    def predict(self,Xnew): # 模型预测
        if self.intercept:
            ones=np.ones(Xnew.shape[0])
            X=np.c_[ones,Xnew]
        else:
            X=Xnew
        Y=X@self.a
        return Y
    def FTest(self,alpha): # F-检验
        yHat=self.predict(self.X)
        Qe=((self.Y-yHat)**2).sum(axis=0)
        yAver=np.mean(self.Y,axis=0)
        U=((yHat-yAver)**2).sum(axis=0)
        Fvalue=(U/self.k_x)/(Qe/(self.n_x-self.k_x-1))
        Falpha=f.isf(alpha,1,self.n_x-self.k_x-1)
        return Fvalue,Falpha,Fvalue>Falpha
    def RTest(self,alpha,i_x=0,i_y=0):
        # 两个元素的R-检验，i_x和i_y是待检验的两变量列指标
        # 默认完成第一个自变量与第一个因变量的R-检验
        # 取出变量
        if i_x==self.k_x:
            X=self.X[:,self.k_x:]
        else:
            X=self.X[:,i_x:i_x+1]
        if i_y==self.k_y:
            Y=self.Y[:,self.k_y:]
        else:
            Y=self.Y[:,i_y:i_y+1]
        # R-检验步骤
        Falpha=f.isf(alpha,1,self.n_x-self.k_x-1)
        Ralpha=np.sqrt(1/(1+(self.n_x-self.k_x-1)/Falpha))
        xAver=np.mean(X,axis=0)
        yAver=np.mean(Y,axis=0)
        Rup1=X-xAver
        Rup2=Y-yAver
        Rup=(Rup1*Rup2).sum()
        Rdown=np.sqrt(((X-xAver)**2).sum()*((Y-yAver)**2).sum())
        R=abs(Rup/Rdown)
        return R,Ralpha,R>Ralpha
    def COV(self,alpha): # 求协方差并进行两两R-检验
        Cov=np.zeros((self.k_x,self.k_y))
        Judgemat=np.zeros((self.k_x,self.k_y))
        for i in range(self.k_x):
            for j in range(self.k_y):
                Cov[i,j],Ralpha,Judgemat[i,j]=self.RTest(alpha,i,j)
        return Cov,Judgemat,Ralpha
                
