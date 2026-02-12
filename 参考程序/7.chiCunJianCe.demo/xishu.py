import cv2
import numpy as numpy


def minAreaRect_r(img):
    ret, img2 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("minshow",img2)
    cv2.waitKey(0)
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


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)
ret,frame = cap.read()
frame = cv2.flip(frame,0)
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


_,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)
_,contours2,hierarchy = cv2.findContours(frame,3,2)
cnt2 = contours2[0]
# 计算边缘图像面积
M2 = cv2.moments(cnt2)
cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])
area2 = cv2.contourArea(cnt2)

area1 = 3.1416*(14.67/2)**2
#area1 = 30.04*30.04

print('area2:%s'%area2)

# 求矩形图像面积
#wf,hf,xf,yf,af = minAreaRect_r(frame)
#area2 = wf*hf


cap.release()
cv2.imshow('frame',frame)
cv2.waitKey(0)

#area1 = 30.04*30.04
ret = (area2/area1)**0.5
print (ret)
