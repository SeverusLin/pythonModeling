import numpy as np
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
