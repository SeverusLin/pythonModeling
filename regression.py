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
            X=np.c_[ones,self.X]
        else:
            X=self.X
        self.a=np.linalg.inv(X.T@X)@(X.T@self.Y)
    def predict(self,Xnew):
        if self.intercept:
            ones=np.ones(Xnew.shape[0])
            X=np.c_[ones,Xnew]
        else:
            X=Xnew
        Y=X@self.a
        return Y
    def Ftest(self,alpha):
        yHat=self.predict(self.X)
        Qe=((self.Y-yHat)**2).sum()
        yAver=self.Y.mean()
        U=((yHat-yAver)**2).sum()
        Fvalue=U/(Qe/(self.X.shape[0]-2))
        Falpha=f.isf(alpha,1,self.X.shape[0]-2)
        return [Fvalue,Falpha]