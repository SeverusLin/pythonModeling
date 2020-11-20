import numpy as np
from PCA import PCA
from REG import MLR
class PCR:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def confirmPCs(self):
        self.pca=PCA(self.X)
        compare,cum=self.pca.SVDdecompose()
        return compare,cum
    def fit(self,PCs):
        T,P=self.pca.PCAdecompose(PCs)
        self.P=P
        self.T=T
        self.mlr=MLR(T,self.Y,False)
        self.mlr.fit()
        #self.A=self.mlr.getCoef()
    def predict(self,Xnew):
        T=np.dot(Xnew,self.P)
        ans=self.mlr.predict(T)
        return ans
    def fTest(self,arfa):
        return self.mlr.Ftest(arfa)


