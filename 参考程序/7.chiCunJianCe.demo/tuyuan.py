"""
图元对象类
"""
import cv2
import tf_geometry as tg
import math


class TuYuan(object):
	"""docstring for TuYuan"""
	def __init__(self,type_,gray=255,point1=[],point2=[],r=0,startage=0,endage=360):
		super(TuYuan, self).__init__()
		self.type_ = type_
		self.gray = gray
		self.point1 = point1
		self.point2 = point2
		self.r = r
		self.startage = startage
		self.endage = endage

		self.points = []
		self.jieGuo = []


	def draw(self,std):
		if self.type_ == 'L':
        #drawL()
			cv2.line(std,(int(self.point1[0]),int(self.point1[1])),(int(self.point2[0]),int(self.point2[1])),self.gray,1)
			return std
		elif self.type_ == 'C':
        #drawA()
			cv2.ellipse(std,(int(self.point1[0]),int(self.point1[1])),(int(2*self.r),int(2*self.r)),0,self.startage,self.endage,self.gray,1)
			return std



	def getPoints(self,img1,img2):
		#if self.type_ == 'C':
			#self.points = np.zeros(img1.shape,np.uint8)
			#self.points = cv2.bitwise_and(img1,img2)
			#cv2.imshow("C",self.points)
			#cv2.waitKey(0)
			#return 

		h,w = img1.shape[:2]
		for i in range(0,h-2):
			for j in range(0,w-2):
				if img1[i,j] == self.gray:
					if img2[i-2,j] >= 100:
						self.points.append((j,i-2))
					if img2[i,j-2] >= 100:
						self.points.append((j-2,i))
					if img2[i-1,j] >= 100:
						self.points.append((j,i-1))
					if img2[i,j-1] >= 100:
						self.points.append((j-1,i))
                        #print(self.points)
					if img2[i,j] >= 100:
						self.points.append((j,i))
					if img2[i,j+1] >= 100:
						self.points.append((j+1,i))
					if img2[i+1,j] >= 100:
						self.points.append((j,i+1))
					if img2[i,j+2] >= 100:
						self.points.append((j+2,i))
					if img2[i+2,j] >= 100:
						self.points.append((j,i+2))
		self.points = sorted(set(self.points),key=self.points.index)


	def getJieGuo(self):
		if self.type_ == 'L':
			dc = tg.TfGeometry(points1 = self.points)
			w,b = dc.Least_squares()
			self.jieGuo = [w,b]
		elif self.type_ == 'C':		
			dc = tg.TfGeometry(points1 = self.points)
			x,y,r = dc.circleLeastFit()
			self.jieGuo = [x,y,r]


class CeLiang(object):
	"""docstring for CeLiang"""
	def __init__(self, jiHeLiang=[],No=[],tuY1=[],tuY2=[],gongChenZhi=[],fuHao=[],jiXian=[],jieGuo=[],ruiJiao=True):
		super(CeLiang, self).__init__()
		self.jiHeLiang = jiHeLiang
		self.tuY1 = tuY1
		self.tuY2 = tuY2
		self.gongChenZhi = gongChenZhi
		self.fuHao = fuHao
		self.jiXian = jiXian
		self.jieGuo = jieGuo
		self.No = No
		self.ruiJiao = ruiJiao


if __name__ == "__main__":
	import numpy as np
	import imgTransform as iT 
	import math
	import tf_geometry as tg 
	import csv


	def minAreaRect_r(img):
		ret, img2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
		# 寻找连通矩形  
		_,contours, hierarchy = cv2.findContours(img2, 3, 2) 
		for contour in contours:  
		# 获取最小包围矩形  
			rect = cv2.minAreaRect(contours[0])  

			# 中心坐标  
			x, y = rect[0]  
			#cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)  

			# 长宽,总有 width>=height  
			width, height = rect[1]  

			# 角度:[-90,0)  
			angle = rect[2]  

			#cv2.drawContours(img, contour, -1, (255, 255, 0), 3)  
			#print ('width=', width, 'height=', height, 'x=', x, 'y=', y, 'angle=', angle ) 
			return width,  height,  x,  y,  angle 


	cap = cv2.VideoCapture(1)
	cap.set(3,1280)
	cap.set(4,1024)
	ret,img1 = cap.read()
	print(img1.shape[:2])
	"""
	mtx = np.loadtxt('mtx.txt')
	dist = np.loadtxt('dist.txt')
	h,  w = img1.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
	img1 = cv2.undistort(img1, mtx, dist, None, newcameramtx)
	"""
	img1 = cv2.flip(img1,0)
	img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	_,img1 = cv2.threshold(img1,80,255,cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)

	img1 = cv2.Canny(img1,100,300)
	cap.release()

	wf,hf,xf,yf,bf = minAreaRect_r(img1)
	print('width=', wf, 'height=', hf, 'x=', xf, 'y=', yf, 'angle=', bf)
	if wf < hf:
		bf = bf+90

	img = np.zeros((1024,1280,1),np.uint8)

	# 读取图元csv文件
	csvfile = open(r'./csv_test.csv','r')
	reader = csv.reader(csvfile)

	#绘制标准图像
	tylist = {}
	for line in reader:
		print (line)
		tylist[str(line[0])+str(line[1])] = TuYuan(type_=line[0],gray=int(line[1])+50,point1=[float(line[2]),float(line[3])],
			point2=[float(line[4]),float(line[5])],r=float(line[4]),startage=float(line[5]),endage=float(line[6]))
		tylist[str(line[0])+str(line[1])] .draw(img)
		print(str(line[0])+str(line[1]))

	#cv2.imshow('img',img)
	#cv2.waitKey(0)

	wi,hi,xi,yi,bi = minAreaRect_r(img)
	print('width_i=', wi, 'height=', hi, 'x=', xi, 'y=', yi, 'angle=', bi)

	af = math.sqrt((wf*hf)*0.9/(wi*hi))
	print('af:%s'%af)


	rows = img.shape[0]
	cols = img.shape[1]
	it = iT.Img(img,rows,cols,center=[int(xf),int(yf)])
	it.SupperMove(beta=math.radians(0),delta_x=int(xf-xi),delta_y=int(yf-yi),factor=1)
	it.Process()
	dst1 = it.dst
	#"""
	ita = iT.Img(dst1,rows,cols,center=[int(xf),int(yf)])
	ita.SupperMove(beta=math.radians(bf),delta_x=0,delta_y=0,factor=1)
	ita.Process()
	dst2 = ita.dst

	itb = iT.Img(dst2,rows,cols,center=[int(xf),int(yf)])
	itb.SupperMove(beta=math.radians(0),delta_x=0,delta_y=0,factor=af)
	itb.Process()
	dst = itb.dst

	dst = cv2.dilate(dst,(9,9),iterations = 1)

	cv2.imshow('dst1',dst1)
	#"""
	cv2.imshow('dst2',dst2)
	cv2.imshow('dst',dst)

	cv2.imshow('img1',img1)
	#cv2.waitKey(0)

	try:
		for key in tylist:
			tylist[key].getPoints(dst,img1)
			print(tylist[key].points)
			if tylist[key].points == []:
				print('%s坐标提取失败！'%tylist[key])
				continue
			tylist[key].getJieGuo()
			print(tylist[key].jieGuo)
			print('***********************')
	except:
		tylist[key].points = []
		print('%s坐标提取失败！'%tylist[key])

	


	#"""
	# 读取测量csv文件
	csvfile = open(r'./csv_ celiang.csv','r')
	reader = csv.reader(csvfile)
	XISHU = 68.84772957664795

	#测量尺寸
	cllist = {}
	for line in reader:
		#try:
		cllist[str(line[0])+str(line[1])] = CeLiang(jiHeLiang=line[0],No=line[1],tuY1=line[2],tuY2=line[3],gongChenZhi=line[4],fuHao=line[5],jiXian=line[6],jieGuo=[],ruiJiao=True)

		if line[0] == 'L2L':#测量两直线的间距
			print ('%s与%s之间的距离：'%(line[2],line[3]))
			w,b = tylist[line[3]].jieGuo
			distance_sum = 0
			for px,py in tylist[line[2]].points:
				distance = (px*w-1*py+b)/((w**2+1)**0.5)
				distance_sum += distance
			cllist[str(line[0])+str(line[1])].jieGuo = (distance_sum/len(tylist[line[2]].points))/XISHU
		elif line[0] == 'P2L':# 测量点到直线距离
			print ('%s与%s之间的距离：'%(line[2],line[3]))
			w,b = tylist[line[3]].jieGuo
			x,y,r = tylist[line[2]].jieGuo
			distance = (x*w-1*y+b)/((w**2+1)**0.5)
			cllist[str(line[0])+str(line[1])].jieGuo = (distance)/XISHU
		elif line[0] == 'P2P': # 测量点到点的距离
			pass
		elif line[0] == 'C': # 测量垂直度
			print('%s与基准%s的垂直度：'%(line[2],line[3]))
			w,b = tylist[line[3]].jieGuo
			w = -1/w
			distance = []
			for px,py in tylist[line[2]].points:
				distance.append(float((px*w-1*py+b)/((w**2+1)**0.5)))
			distance = sorted(set(distance),key=distance.index)
			cllist[str(line[0])+str(line[1])].jieGuo = ((distance[0]-distance[-1])/XISHU)
		elif line[0] == 'P': # 测量平行度
			print('%s与基准%s的平行度：'%(line[2],line[3]))
			w,b = tylist[line[3]].jieGuo
			distance = []
			for px,py in tylist[line[2]].points:
				distance.append(float((px*w-1*py+b)/((w**2+1)**0.5)))
			distance = sorted(set(distance),key=distance.index)
			cllist[str(line[0])+str(line[1])].jieGuo = ((distance[0]-distance[-1])/XISHU)
		elif line[0] == 'Z': # 测量直线度
			print('%s的直线度：'%line[2])
			w,b = tylist[line[2]].jieGuo
			distance = []
			for px,py in tylist[line[2]].points:
				distance.append(float((px*w-1*py+b)/((w**2+1)**0.5)))
			distance = sorted(set(distance),key=distance.index)
			cllist[str(line[0])+str(line[1])].jieGuo = ((distance[0]-distance[-1])/XISHU)
		elif line[0] == 'J': # 测量角度
			print('%s与基准%s的角度：'%(line[2],line[3]))
			k1,b1 = tylist[line[2]].jieGuo
			k2,b2 = tylist[line[3]].jieGuo
			cllist[str(line[0])+str(line[1])].jieGuo = round((math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5),3)
		elif line[0] == 'ZJ':# 测量圆的直径
			print ('%s的直径：'%line[2])
			x,y,r = tylist[line[2]].jieGuo
			cllist[str(line[0])+str(line[1])].jieGuo = r*2/XISHU

		elif line[0] == 'C2C':# 测量圆心距、同心度
			print ('%s与%s的圆心距：'%(line[2],line[3]))
			x1,y1,r1 = tylist[line[2]].jieGuo
			x2,y2,r2 = tylist[line[3]].jieGuo
			d = math.sqrt((x2-x1)**2+(y2-y1)**2)
			cllist[str(line[0])+str(line[1])].jieGuo = d*2/XISHU
		else:
			print('没有这个测量项目！')

		print(cllist[str(line[0])+str(line[1])].jieGuo)


	cv2.waitKey(0)
