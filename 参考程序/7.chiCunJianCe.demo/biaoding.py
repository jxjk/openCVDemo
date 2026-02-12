import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

import imgTransform as iT
import tf_geometry as tg 



# 生成标定块描述文件
# L1 20.0 20.0 50.25 50.5 #LINE((20.0,20.0),(50.25,50.5),1)
# A2 50.5 25.4 0.50 0 82  #ARC((50.5,25.4),0.50,0,82,2)
# 读取文件 type_ = L no. = 1 X1 = 20.0 Y1 = 20.0 X2 = 50.25 Y2 = 50.25 
#          type_ = A no. = 2 x1 = 50.5 y1 = 25.4 r = 0.5 start = 0 end = 82 
def readfile(pixel_mm=5.0,shape=(600,480,1)):
    type_ = 'L'
    No1 = 60 

    x1 = 0.0
    y1 = 0.0
    x2 = 6.06
    y2 = 30.04

    No2 = 70
    No3 = 80
    No4 = 90
    
    img = np.zeros(shape,np.uint8)
    
    #drawstd(std=img,type_=type_,No=No,x1=x1*pixel_mm,y1=y1*pixel_mm,x2=x2*pixel_mm,y2=y2*pixel_mm)
    drawstd(std=img,type_=type_,No=No1,x1=x1*pixel_mm,y1=y1*pixel_mm,x2=x1*pixel_mm,y2=y2*pixel_mm)
    drawstd(std=img,type_=type_,No=No3,x1=x2*pixel_mm,y1=y1*pixel_mm,x2=x2*pixel_mm,y2=y2*pixel_mm)

    drawstd(std=img,type_=type_,No=No2,x1=x1*pixel_mm,y1=y1*pixel_mm,x2=x2*pixel_mm,y2=y1*pixel_mm)
    drawstd(std=img,type_=type_,No=No4,x1=x1*pixel_mm,y1=y2*pixel_mm,x2=x2*pixel_mm,y2=y2*pixel_mm)
    #drawstd(std=img,type_=type_2,No=No,x1=int(x2*pixel_mm),y1=int(y2*pixel_mm),r=int(R*pixel_mm),start=start,end=end)
    #drawstd(std=img,type_=type_2,No=No2,x1=int(x2*pixel_mm),y1=int(y2*pixel_mm),r=int(r*pixel_mm),start=start,end=end)
    rows,cols = img.shape


    #M = np.float32([[1,0,100],[0,1,100]])
    #img = cv2.warpAffine(img,M,(cols,rows))

    _,contours,hierarchy = cv2.findContours(img,3,2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(cnt)
    cv2.imshow('std',img)
    # cv2.waitKey(0)
    return area,img,cx,cy
    


# 标定块位图
def drawstd(std,type_,No=0,x1=0,y1=0,x2=0,y2=0,r=0,start=0,end=0):
    if type_ == 'L':
        #drawL()
        cv2.line(std,(int(x1),int(y1)),(int(x2),int(y2)),No,1)
        return std
    elif type_ == 'A':
        #drawA()
        cv2.ellipse(std,(int(x1),int(y1)),(2*r,2*r),0,start,end,No,1)
        return std


    # 生成掩模函数
class CMask:
    def __init__(self,img,kernel,No):
        self.img = img#.copy()
        self.kernel = kernel
        self.No = No
        self.mask = np.zeros(self.img.shape,np.uint8)
        self.createMask()


    def createMask(self):
        for j in range(rows):
            for i in range(cols):
                if self.img[j,i] == self.No:
                    self.mask[j,i] = 255 
                else:
                    self.mask[j,i] = 0
        self.mask = cv2.dilate(self.mask,self.kernel,iterations = 1)


class FindPoints:
    def __init__(self,std,img,No):
        self.std = std
        self.img = img
        self.No = No
        self.points = []
        self.ps = []
        self.findP()

    
    def findP(self):
        h,w = self.std.shape[:2]
        for i in range(0,h-1):
            for j in range(0,w-1):
                if self.std[i,j] == self.No:
                    if self.img[i-1,j] >= 0:
                        self.points.append((j,i-1))

                    if self.img[i,j-1] >= 0:
                        self.points.append((j-1,i))
                        #print(self.points)
                    if self.img[i,j] >= 0:
                        self.points.append((j,i))
                    if self.img[i,j+1] >= 0:
                        self.points.append((j+1,i))
                    if self.img[i+1,j] >= 0:
                        self.points.append((j,i+1))
        self.ps = sorted(set(self.points),key=self.points.index)




# 标定块采集图像:
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)
ret,frame = cap.read()
frame = cv2.flip(frame,0)
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
h,w = frame.shape
print(frame.shape)
frame = cv2.resize(frame,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)
print(frame.shape)
# 边缘图像
_,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)
_,contours2,hierarchy = cv2.findContours(frame,3,2)
cnt2 = contours2[0]
# 计算边缘图像面积
M2 = cv2.moments(cnt2)
cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])
area2 = cv2.contourArea(cnt2)
print('area2:%s'%area2)
edges = cv2.Canny(frame,100,300)

cap.release()







# 匹配
# pixel_mm初始化
# 6.7
pixel_mm = 6.7


# 计算标准图像面积
# 标准图像面积大于边缘图像面积的101%，loop 
# 标准图像面积小于边缘图像面积的99%，loop
# 其它，返回标准图像
while True:
    area1,img,cx1,cy1 = readfile(pixel_mm,shape=frame.shape[:])
    print('area1:%s'%area1)
    print(cx1,cy1)
    if ((area2*0.98 <= area1) and (area1 <= area2*1.1)):
        break 
    elif (area2*0.98 <= area1):
        pixel_mm -= 0.004
    elif (area1 <= area2*1.1):
        pixel_mm += 0.009
# """
# 将标准图像进行平移、旋转，使之与边缘图像重合
# 定义仿射矩阵
rows,cols = img.shape
print(rows)
print(pixel_mm)


def minAreaRect_r(img):
    ret, img2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    #cv2.imshow("minshow",img2)
    #cv2.waitKey(0)
    # 寻找连通矩形  
    _,contours, hierarchy = cv2.findContours(img2, 3, 2)  

    for contour in contours:  
    # 获取最小包围矩形  
        rect = cv2.minAreaRect(contours[0])  
    
        # 中心坐标  
        x, y = rect[0]  
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)  
    
        # 长宽,总有 width>=height  
        width, height = rect[1]  
    
        # 角度:[-90,0)  
        angle = rect[2]  
        
        #cv2.drawContours(img, contour, -1, (255, 255, 0), 3)  
        #print ('width=', width, 'height=', height, 'x=', x, 'y=', y, 'angle=', angle ) 
        return width,  height,  x,  y,  angle 


cv2.imshow('frame',edges)
#cv2.waitKey(0)
#"""
#wi,hi,xi,yi,ai = minAreaRect_r(img)
#print('width=', wi, 'height=', hi, 'x=', xi, 'y=', yi, 'angle=', ai)
wf,hf,xf,yf,af = minAreaRect_r(frame)
print('width=', wf, 'height=', hf, 'x=', xf, 'y=', yf, 'angle=', af)

#Mro = cv2.getRotationMatrix2D((xi,xi),-af,1)
#img = cv2.warpAffine(img,Mro,(0,0))
#cv2.imshow('std_move1',img)
#cv2.waitKey(0)
#"""        self.src=image #原始图像
        #self.rows=rows #原始图像的行
        #self.cols=cols #原始图像的列
        #self.center=center #旋转中心，默认是[0,0]

rows = img.shape[0]
cols = img.shape[1]

ita = iT.Img(img,rows,cols,[cx1,cy1])

if wf > hf:
    #ita.Rotate(((af)*2*3.1415/360))
    ita.RotMove(((90+af)*2*3.1415/360),delta_x=xf-cx1,delta_y=cy1-yf)
else:
    #ita.Rotate(((90+af)*2*3.1415/360))
    ita.RotMove(((af)*2*3.1415/360),delta_x=xf-cx1,delta_y=cy1-yf)
ita.Process()

dst = ita.dst


"""

pts1 = np.float32([[xi+math.sin(2*3.14/360*ai)*(hi/2),yi+math.cos(2*3.14/360*ai)*(hi/2)],[xi,yi],[xi+((wi/2)*math.cos(2*3.14/360*ai)),yi+math.sin(2*3.14/360*ai)*(wi/2)]])
pts2 = np.float32([[xf+math.sin((2*3.14/360*(90+af)))*(hi/2),yf+math.cos((2*3.14/360*(90+af)))*(hi/2)],[xf,yf],[xf+((wi/2)*math.cos((2*3.14/360*(90+af)))),yf-math.sin((2*3.14/360*(90+af)))*(wi/2)]])


M = cv2.getAffineTransform(pts1,pts2)

#M = np.float32([[1,0,(xf-xi)],[0,1,(yf-yi)]])
"""
# 仿射变换
#dst = cv2.warpAffine(img,M,(cols,rows))
#wi,hi,xi,yi,ai = minAreaRect_r(dst)
#print('width1=', wi, 'height1=', hi, 'x=', xi, 'y=', yi, 'angle=', ai)
#print(dst)
cv2.imshow('std_moveR',dst)
#cv2.imshow('std_move2',dst)
cv2.waitKey(0)

"""
# 生成图元掩模

dst1=dst.copy()
dst2 = dst.copy()

a = CMask(dst1,No=70,kernel=np.ones((5,5),np.uint8))
#a.createMask()
mask1 = a.mask 
cv2.imshow('mask1',mask1)
cv2.waitKey(0)
"""

b = FindPoints(dst,edges,70)
print("b:\n")
print(b.ps)

c = FindPoints(dst,edges,90)
print("c:\n")
print(c.ps)

"""
b = CMask(dst2,No=90,kernel=np.ones((5,5),np.uint8))
#b.createMask()        
mask2 = b.mask
cv2.imshow('mask2',mask2)
cv2.waitKey(0)


# dst = cv2.add(edges,dst)

# 以标准图像个图元为掩模，使用圆域寻找边缘图像对应的像素点
tuyuan1 = cv2.bitwise_and(mask1,edges)
tuyuan2 = cv2.bitwise_and(mask2,edges)


# 将图元转换为np数组
def qiuZuoBiao(img):
    h,w = img.shape[0:2]
    ps = []
    for i in range(0,h):
        for j in range(0,w):
            if img[i,j] > 0:
                ps.append([j,i])
    return ps


tuyuan1_p= qiuZuoBiao(tuyuan1)
tuyuan2_p = qiuZuoBiao(tuyuan2)
print(tuyuan1_p)
print('2:\n')
print(tuyuan2_p)
"""

#a = tg.TfGeometry(points1= [[0,0],[1,1],[2,2]])

#aw,ab = a.getLine()
#print('aw=%s ab=%s'%(aw,ab))

#d = tg.TfGeometry(points1 = tuyuan1_p,points2 = tuyuan2_p)
d = tg.TfGeometry(points1 = b.ps,points2 = c.ps)
dd = d.getP2L()
print(dd/13.493593294936426)

# 计算标定系数
# 13.395565355877936 13.4676321732007 13.491904288393124 13.608972276080344 13.336119310971126
# 13.323753316635251 13.410447668384437 13.57058733025309 13.400225305818376

# """
# 标定结束
#cv2.imshow('tuyuan1',tuyuan1)
#cv2.imshow('tuyuan2',tuyuan2)
cv2.waitKey(0)

#"""
