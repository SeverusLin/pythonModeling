#双变量线性回归
import numpy as np
from scipy.stats import f
class MLR:
    def __init__(self,X,Y,intercept=True):
        self.X = X
        self.Y = Y
        self.intercept = intercept
    def fit(self):
        if self.intercept:
            ones=np.ones(self.X.shape[0])
            A=np.c_[ones,self.X]
        else:
            A=self.X
        self.a=np.linalg.inv(A.T@A)@(A.T@self.Y)
    def predict(self,Xnew):
        if self.intercept:
            ones=np.ones(Xnew.shape[0])
            X=np.c_[ones,Xnew]
        else:
            X=Xnew
        Y=X@self.a
        return Y
    def FTest(self,alpha):
        yHat=self.predict(self.X)
        Qe=((self.Y-yHat)**2).sum(axis=0)
        yAver=np.mean(self.Y,axis=0)
        U=((yHat-yAver)**2).sum(axis=0)
        n,k=self.X.shape

        Fvalue=(U/k)/(Qe/(n-k-1))
        Falpha=f.isf(alpha,1,n-k-1)
        return Fvalue,Falpha,Fvalue>Falpha
    def RTest(self,alpha):# 两个元素的检验
        Falpha=f.isf(alpha,1,self.X.shape[0]-2)
        Ralpha=np.sqrt(1/(1+(self.X.shape[0]-2)/Falpha))
        xAver=np.mean(self.X)
        yAver=np.mean(self.Y)
        Rup1=self.X-xAver
        Rup2=self.Y-yAver
        Rup=(Rup1*Rup2).sum()
        Rdown=np.sqrt(((self.X-xAver)**2).sum()*((self.Y-yAver)**2).sum())
        R=abs(Rup/Rdown)
        return R,Ralpha,R>Ralpha