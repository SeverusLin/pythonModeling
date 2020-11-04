import numpy as np
import numpy.random as npr
class Individual:
    _n=0
    eval=0.0
    chromsome=None
    def __init__(self,n):
        self._n=n
        self.chromsome=npr.rand(n)
    def crossover(self,another):
        startPos = npr.randint(self._n) #交叉的起始位置
        jeneLength = npr.randint(self._n)+1 # //交叉的长度
        son1 = Individual(self._n)
        son2 = Individual(self._n)

        son1.chromsome[0:startPos]=self.chromsome[0:startPos]
        son2.chromsome[0:startPos]=another.chromsome[0:startPos]
        
        endpos=startPos+jeneLength
        son1.chromsome[startPos:endpos]=another.chromsome[startPos:endpos]
        son2.chromsome[startPos:endpos]=self.chromsome[startPos:endpos]
        son1.chromsome[endpos:]=self.chromsome[endpos:]
        son2.chromsome[endpos:]=another.chromsome[endpos:]
        return son1,son2
    def mutation(self,learnRate):
        son = Individual(self._n)
        son.chromsome=self.chromsome.copy()
        mutationPos =npr.randint(self._n)#;//变异的位置
        #产生一个-0.5-0.5之间的随机小数
        temp = npr.rand()-0.5
        son.chromsome[mutationPos] += learnRate * temp
        return son

        
class NGA:
    population=[]
    dimension=1
    bestPos=worstPos=0
    mutationProb=10
    crossoverProb=90
    maxIterTime=1000
    evalFunc=None
    arfa =1.0
    popu=2
    def __init__(self,popu, dimension,crossoverProb,mutationProb,maxIterTime,evalFunc):
        for i in range(popu):
            oneInd=Individual(dimension)
            oneInd.eval=evalFunc(oneInd.chromsome)
            self.population.append(oneInd)
            
        self.crossoverProb=crossoverProb
        self.mutationProb=mutationProb
        self.maxIterTime=maxIterTime
        self.evalFunc=evalFunc
        self.popu=popu
        self.dimension=dimension
    
    #找最好的个体位置
    def findBestWorst(self):
        self.population.sort(key=lambda o:o.eval)
        self.bestPos=0
        self.worstPos=self.popu-1
       #交叉操作 
    def crossover(self):
        fatherPos=npr.randint(0,self.popu)
        motherPos=npr.randint(0,self.popu)
        while motherPos == fatherPos:
            motherPos = npr.randint(0,self.popu)
        father = self.population[fatherPos]
        mother = self.population[motherPos]
        son1,son2=father.crossover(mother)
        
        son1.eval = self.evalFunc(son1.chromsome) #;// 评估第一个子代
        son2.eval = self.evalFunc(son2.chromsome)
        self.findBestWorst()
        
        if son1.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son1
        self.findBestWorst()
        if son2.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son2
            
    def mutation(self):
        father = self.population[npr.randint(self.popu)]
        son=father.mutation(self.arfa)
        son.eval = self.evalFunc(son.chromsome)
        self.findBestWorst()
        if son.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son
            
    def solve(self):
        shrinkTimes = self.maxIterTime / 10 
        #//将总迭代代数分成10份
        oneFold = shrinkTimes #;//每份中包含的次数
        i = 0
        while i < self.maxIterTime:
            print(i,"---",self.maxIterTime)
            if i == shrinkTimes:
                self.arfa =self.arfa / 2.0
            #经过一份代数的迭代后，将收敛参数arfa缩小为原来的1/2，以控制mutation
                shrinkTimes += oneFold  #;//下一份到达的位置
            for  j in range(self.crossoverProb):
                self.crossover()
            for  j in range(self.mutationProb):
                self.mutation()
            print("solution:",self.population[self.bestPos].chromsome)
            print("func value:",self.population[self.bestPos].eval)
            i=i+1
            
    def getAnswer(self):
        self.findBestWorst()
        return self.population[0].chromsome



