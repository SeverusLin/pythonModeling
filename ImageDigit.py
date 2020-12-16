import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os # 创建目录时用

class ImageDigit:
    def __init__(self,im):
        self.im=im
    def split(self):  # im为去除背景后的整个图像
        assert self.im.mode == '1'
        im_arr = 1-np.array(self.im)  # 黑点1，其他0
        xmax, ymax = self.im.size
        row_arr = im_arr.sum(axis=1) # 行有多少个黑点
        col_arr = im_arr.sum(axis=0)  # 每列有多少个黑点
        row_arr = row_arr[1:]  # 第一行去掉
        row_boundary_list = []
        for i in range(len(row_arr)-1):  # 最后一行也去掉
            if row_arr[i] == 0 and row_arr[i+1] != 0 or row_arr[i] != 0 and row_arr[i+1] == 0:
                row_boundary_list.append(i+1)
                
        yl = [ row_boundary_list[i:i+2] for i in range(0,len(row_boundary_list),2) ]

        column_boundary_list = []
        for i,x in enumerate(col_arr[:-1]):
            if (col_arr[i] == 0 and col_arr[i+1] != 0) or col_arr[i] != 0 and col_arr[i+1] == 0:
                column_boundary_list.append(i+1)
        xl = [ column_boundary_list[i:i+2] for i in range(0,len(column_boundary_list),2) ]
    
        xs = [i[0] for i in xl]+[xmax]
        ys = [i[0] for i in yl]+[ymax]
        
        print('xs=',xs)  # 输出列边界
        print('ys=',ys)  # 输出行边界
        if not os.path.exists('./result_22/'): # 存储分割后小图像位置
            os.makedirs('./result_22/')
         
        digits=[]  # 分割的小图像存储位置
        for i, x in enumerate(xs):
            if i + 1 >= len(xs):
                break
            oneDigit=[]
            for j, y in enumerate(ys):
                if j + 1 >= len(ys):
                    break
                box = (x, y, xs[i + 1], ys[j + 1])  # 一个字符所在图像的位置
                print(box)
                
                t = self.im.crop(box).copy()  # 将字符挖出来
                t_arr = 1-np.array(t)
                print('ttt=',t_arr.sum())
                #t.show()
                
               
                if t_arr.sum()>25:  # 大于25像素，认为是一个有效数字
                    t=self.centered(t)
                    oneDigit.append(t)
               # 
            if len(oneDigit)>=1:
                digits.append(oneDigit)
        self.digits=digits
        # 可以不输出
        for i in range(len(digits)):  # 输出分割出来的小图像
            for j in range(len(digits[i])):
                t=digits[i][j]
                t.save("./result_22/" + str((i + 1)%10) + "_" + str(j) + ".bmp")
        return digits


    def centered(self,im):
        im = im.convert("1")
        w, h = im.size
        xrow = []
        ycol = []
        for i in range(w):
            for j in range(h):
                pixel = im.getpixel((i, j))  # 返回某一点的像素值
                if (pixel < 1):
                    xrow.append(i)
                    ycol.append(j)
        if (len(xrow)==0):
            print("find")
            im.show()
            
        xLength = max(xrow) - min(xrow) + 1
        yLength = max(ycol) - min(ycol) + 1  # 得到字符的长度宽度
        box = (min(xrow), min(ycol), max(xrow) + 1, max(ycol) + 1)
        t = im.crop(box).copy()  # 从当前的图像中返回一个矩形区域的拷贝。变量box是一个四元组，定义了左、上、右和下的像素坐标
        w, h = t.size
        
        ma=max(w,h)
        
        xStart = (ma - xLength) // 2
        yStart = (ma - yLength) // 2  # 居中
        bg = Image.new('1', (ma, ma), 'white')
        bg.paste(t, (xStart, yStart))  # 将一张图粘贴到另一张图像上
        t = bg.resize((32, 32),Image.ANTIALIAS)  # 缩放到32*32
        return t
                       

    def histShow(self):
        img=np.array(self.im.convert('L'))
        plt.figure("hist")
        arr=img.flatten()
        n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)  
        plt.show()
        
    def convert_to_bw(self,threshold):
        im = self.im.convert("L")  # 转换为灰度图像
        im = im.point(lambda x: 255 if x > threshold else 0) #代表数字的像素值置0，黑色
        im = im.convert('1') # 黑白二值图像
        self.im=im
        im.show()

    def to_32_32(self):
        if not os.path.exists('./result_32/'): # 存储分割后小图像位置
            os.makedirs('./result_32/')

        digits32_32=[]
        for ii in range(len(self.digits)):
            row=self.digits[ii]
            oneDigit=[]
            for jj in range(len(row)):
                im=row[jj]
                im = im.convert("L")
                w, h = im.size
                xrow = []
                ycol = []
                for i in range(w):
                    for j in range(h):
                        pixel = im.getpixel((i, j))  # 返回某一点的像素值
                        if (pixel < 1):
                            xrow.append(i)
                            ycol.append(j)
                xLength = max(xrow) - min(xrow) + 1
                yLength = max(ycol) - min(ycol) + 1  # 得到字符的长度宽度
                box = (min(xrow), min(ycol), max(xrow) + 1, max(ycol) + 1)
                t = im.crop(box).copy()  # 从当前的图像中返回一个矩形区域的拷贝。变量box是一个四元组，定义了左、上、右和下的像素坐标
                
                xStart = (32 - xLength) // 2
                yStart = (32 - yLength) // 2  # 居中
                bg = Image.new('L', (32, 32), 'white')
                bg.paste(t, (xStart, yStart))  # 将一张图粘贴到另一张图像上
                bg.save("./result_32/" + str((ii+1)%10) + "_" + str(jj) + "_32_32.bmp")
                oneDigit.append(bg)
            digits32_32.append(oneDigit)
        self.digits32_32=digits32_32
        
    def featureExtract_new(self):
        netTrainDataInput = []  # 存储输入数据，X
        netTrainDataoutput = []  # 存储输出点的真值，Y
        for kk in range(len(self.digits32_32)):
            row=self.digits32_32[kk]
            outNode = [0.0] * 10
            outNode[(kk+1)%10] = 1.0  # 训练集的函数真值，每种模式对应的位置数字为1
            for kkk in range(len(row)):
                im = row[kkk]#获取每一幅图片
                im = im.convert("1")#图片二值化，那么接下来可以直接利用向量化操作得出features
                features = []  # 将32*32的小图像分割成256个2*2的范围，统计每个2*2的小方框中，像素值为0的个数，由此将每个字符表达为256个数字组成的向量
                imdata = 1-np.array(im) # 此时imdata是一个布尔型矩阵，空白部分为True,黑色部分为False
                for i in range(1,17):
                    for j in range(1,17): #从1取到16，而不是0开始
                        t = imdata[2*(i-1):2*i,2*(j-1):2*j] #只好对矩阵循环分片
                        features.append(t.sum())  #至此一个图片的features已生成
                netTrainDataInput.append(features)#添加的是每一幅图片的feature值
                netTrainDataoutput.append(outNode)
        X = np.array(netTrainDataInput) # 建模用的
        y = np.array(netTrainDataoutput)
        return X,y      

            

    def featureExtract(self):
        netTrainDataInput = []  # 存储输入数据，X
        netTrainDataoutput = []  # 存储输出点的真值，Y
        for kk in range(len(self.digits32_32)):
            row=self.digits32_32[kk]
            outNode = [0.0] * 10
            outNode[(kk+1)%10] = 1.0  # 训练集的函数真值，每种模式对应的位置数字为1
            for kkk in range(len(row)):
                im = row[kkk]
                im = im.convert("1")
                y = np.array([0, 1])
                features = []  # 将32*32的小图像分割成256个2*2的范围，统计每个2*2的小方框中，像素值为0的个数，由此将每个字符表达为256个数字组成的向量
                for j in range(16):
                    x = np.array([0, 1])
                    for i in range(16):
                        box = (x[0], y[0], x[1] + 1, y[1] + 1)  # 开区间
                        t = im.crop(box).copy()
                        count = 0.0
                        for ii in range(2):
                            for jj in range(2):
                                pixel = t.getpixel((ii, jj))
                                if (pixel < 1):
                                    count += 1.0
                        features.append(count)
                        x = x + 2  # np.array+2整个矩阵向右平移
                    y = y + 2
            
                netTrainDataInput.append(features)
                netTrainDataoutput.append(outNode)
        X = np.array(netTrainDataInput) # 建模用的
        y = np.array(netTrainDataoutput)
        return X,y
           
    def featureExtract1(self):
        netTrainDataInput = []  # 存储输入数据，X
        netTrainDataoutput = []  # 存储输出点的真值，Y
        for kk in range(len(self.digits32_32)):
            row=self.digits32_32[kk]
            outNode = [0.0] * 10
            outNode[kk] = 1.0  # 训练集的函数真值，每种模式对应的位置数字为1
            for kkk in range(len(row)):
                im = row[kkk]
                im = im.convert("L")
                imgData=np.array(im)
                features = imgData.flatten()
                
            
                netTrainDataInput.append(features)
                netTrainDataoutput.append(outNode)
        X = np.array(netTrainDataInput) # 建模用的
        y = np.array(netTrainDataoutput)
        return X,y
