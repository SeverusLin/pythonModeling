import numpy as np  
import matplotlib.pyplot as plt
class AvgStdBar:
    def __init__(self,X,indx,className):
        self.X=X
        self.indx=indx
        self.className=className
    def pltBar(self):
        X1=self.X
        
        X1=X1[:,self.indx]
        
        avg1=X1.mean(axis=0)
        std1=X1.std(axis=0)
        fig, ax = plt.subplots()
        ind = np.arange(X1.shape[-1])
        width = 0.35
        rects = ax.bar(ind, avg1, width, color='r', yerr=std1, align='center')
        
        # 设置add some text for labels, title and axes ticks
        ax.set_ylabel('avg. and std')
        ax.set_xlabel('Feature No.')
        ax.legend( (rects[0], ), (self.className,) )
        ax.set_xticks(ind)
        ax.set_xticklabels( self.indx )
        
        for rect in rects:
            height = rect.get_height() 
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),ha='center', va='bottom')
        plt.show()
 