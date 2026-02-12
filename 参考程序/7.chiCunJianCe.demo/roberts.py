#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#using namespace std;
#using namespace cv;
import cv2 
import numpy as np
import math


def Roberts(src):	#//函数声明
#/*定义Roberts算子函数*/
	dst=src.copy();
	nWidth =dst.shape[1];				#//列数
	nHeight =dst.shape[0];				#//行数
	#pixel=[];					#//定义数组用于保存两行4个像素的值
	for j in range(nHeight-1):          	#//行
		for i in range(nWidth-1):      #//
		
			#//生成Roberts算子
			a = dst[j,i];
			b = dst[j,i+1];
			c = dst[j+1,i];
			d = dst[j+1,i+1];
			#print(a,b,c,d)

			dst[j,i] = int(math.sqrt((a - d) * (a - d)+(b - c) * (b - c) ) );
			#dst[j,i] = dst[j,i] + 100
			#if dst[j,i] > 255:
				#dst[j,i] = 255
			#elif dst[j,i] < 0:
				#dst[j,i] = 0
			#print(dst[j,i])
			#
	#cv2.imshow("dst1",dst)
	return dst

if __name__ == "__main__":
	srcImage=cv2.imread(r"C:\Users\00597\Pictures\images\daiLun1.jpg",0);	
	cv2.imshow("src",srcImage);
	dst1 = np.zeros(srcImage.shape,np.uint8);
	dst1 = Roberts(srcImage);			#//调用Roberts函数
	cv2.imshow("dst",dst1);
	cv2.waitKey(0);
	cv2.destroyAllWindows();

