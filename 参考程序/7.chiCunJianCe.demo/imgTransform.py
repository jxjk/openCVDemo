import cv2
import math
import numpy as np

class Img:
    """图像平移、旋转、缩放、镜像，含超级变换方法"""
    def __init__(self,image,rows,cols,center=[0,0]):
        self.src=image #原始图像
        self.rows=rows #原始图像的行
        self.cols=cols #原始图像的列
        self.center=center #旋转中心，默认是[0,0]


    def Move(self,delta_x,delta_y):      #平移
        #delta_x<0左移，delta_x>0右移
        #delta_y<0上移，delta_y>0下移
        self.transform=np.array([[1,0,-delta_y],[0,1,-delta_x],[0,0,1]])


    def Zoom(self,factor):               #缩放
        #factor<1表示缩小；factor>1表示放大
        factor = 1/factor
        self.transform=np.array([[factor,0,0],[0,factor,0],[0,0,1]])


    def Horizontal(self):                #水平镜像
        self.transform=np.array([[1,0,0],[0,-1,self.cols-1],[0,0,1]])


    def Vertically(self):                #垂直镜像
        self.transform=np.array([[-1,0,self.rows-1],[0,1,0],[0,0,1]])


    def Rotate(self,beta):               #旋转
        #beta<0表示时顺针旋转；beta>0表示逆时针旋转
        self.transform=np.array([[math.cos(beta),-math.sin(beta),0],
                                 [math.sin(beta), math.cos(beta),0],
                                 [    0,              0,         1]])


    def RotMove(self,beta,delta_x,delta_y):               #旋转+平移
        #beta<0表示顺时针旋转；beta>0表示逆时针旋转
        #delta_x<0左移，delta_x>0右移
        #delta_y<0上移，delta_y>0下移
        self.transform=np.array([[math.cos(beta),math.sin(beta),-math.cos(beta)*(-delta_y)+math.sin(beta)*(-delta_x)],
                                 [math.sin(beta), math.cos(beta),math.cos(beta)*(-delta_x)-math.sin(beta)*(-delta_y)],
                                 [0,0,1]])


    def SupperMove(self,beta=0,delta_x=0,delta_y=0,factor=1):               #旋转+平移+缩放
        #beta<0表示顺时针旋转；beta>0表示逆时针旋转
        #delta_x<0左移，delta_x>0右移
        #delta_y<0上移，delta_y>0下移
        #factor<1表示缩小；factor>1表示放大
        factor = 1/factor
        self.transform=np.array([[factor*math.cos(beta),-factor*math.sin(beta),math.cos(beta)*(-delta_y)+math.sin(beta)*(-delta_x)],
                                 [factor*math.sin(beta), factor*math.cos(beta),math.cos(beta)*(-delta_x)-math.sin(beta)*(-delta_y)],
                                 [0,0,1]])


    def Process(self):
        self.dst=np.zeros((self.rows,self.cols),dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos=np.array([i-self.center[1],j-self.center[0],1])
                [x,y,z]=np.dot(self.transform,src_pos)
                x=int(x)+self.center[1]
                y=int(y)+self.center[0]

                if x>=self.rows or y>=self.cols or x<0 or y<0:
                    self.dst[i][j]=0
                else:
                    self.dst[i][j]=self.src[x][y]


if __name__=='__main__':
    src=cv2.imread(r'C:\Users\Public\Pictures\Sample Pictures\test.png',0)
    rows = src.shape[0]
    cols = src.shape[1]
    cv2.imshow('src', src)

    img=Img(src,rows,cols,[int(cols/2+100),int(rows/2)])# y,x
    #img.Horizontal()
    #img.Vertically() #镜像
    #img.Process()
    
    #img.Rotate(-math.radians(180)) #旋转
    #img.Process()
    #img.Move(-100,-50) #平移(y,x)
    #img.Process()
    #img.Zoom(0.5) #缩放
    #img.RotMove(math.radians(30),0,-50)
    img.SupperMove(beta=math.radians(-30),delta_x=0,delta_y=0,factor=0.5)
    img.Process()
    
    cv2.imshow('dst', img.dst)
    cv2.waitKey(0)