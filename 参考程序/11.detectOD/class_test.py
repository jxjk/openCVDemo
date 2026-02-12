# -*- coding: utf-8 -*-
# 生产技术部 蒋小军
# 2018.9.5
# P组同步带轮齿数图像检测装置程序
#
# dailunshuchi.py
##########################################################################################
"""
利用openCV计算带轮齿的数量
条件：背光照明，暗视野
方法：原图》滤波处理》二值化》区域填充》闭运算（100圆）》求闭运算与区域填充的差值》求差值的联通区域个数
输入：含1个带轮的图像
输出：带轮的齿数

所有者：蒋小军 jxjk@163.com
用于同步带轮齿数检测。
2018.10.8
"""
import tkinter as tk
import tkinter.font as tkFont
import tkinter.ttk as ttk
import imutils
import cv2 
import numpy 
import re
import time
import csv
import os
import gc
import pandas as pd
import pypyodbc
import threading
import gxipy as gx
from PIL import Image

macroFile = r"C:\MACRO\macro.txt"
macroCopy = r"macro.txt"
timeSave = 0
PXCEL = 0.098 
"""
class Access(object):
    def __init__(self,db_name,password=""):
        try:
            # str1 = r'Driver={Microsoft Access Driver (*.mdb)};PWD' + password + ';DBQ=' + db_name = r'./data/toolData.mdb'
            str2 = u'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + db_name
            print(str2)
            self.conn = pypyodbc.win_connect_mdb(str2)
        except Exception as e:
            print(e)
        else:
            print('连接成功')
            self.cur = self.conn.cursor()


    def close(self):
        self.cur.close()
        self.conn.close()


    def add(self,sql = 'insert into tooltb values("S2M12-16",33.000,0.776),("S2M17-24",33.001,0.778)'): # 增
        try:
            self.cur.execute(sql)
            self.conn.commit()
            return True
        except:
            return False



    def rem(self,sql = 'delete from tooltb where name="S2M17-24"'):  # 删
        try:
            self.cur.execute(sql)
            self.conn.commit()
            return True
        except:
            return False


    def modi(self,sql =  'update tooltb set diameter=33.534 where name="S2M12-16"'): # 改
        try:
            self.cur.execute(sql)
            self.conn.commit()
            return True
        except:
            return False


    def sel(self,sql = 'select * from tooltb'): # 查
        try:
            self.cur.execute(sql)
            return self.cur.fetchall() 
        except:
            return []
"""

class Detect:
    def __init__(self,img):
        """
        """
        # 原始图像信息
        self.ori_img = img
        # self.gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
        self.gray = img 
        # 获取原始图像行列
        rows, cols = self.ori_img.shape[:2]
        # 工作图像
        # self.work_img = cv2.resize(self.ori_img, (int(cols / 2), int(rows / 2)))
        # self.work_gray = cv2.resize(self.gray, (int(cols / 2), int(rows / 2)))
        self.work_img = self.ori_img
        self.work_gray = self.gray

    # 形态学处理
    def get_good_thresh_img(self,img):
        """
        """
        # 使用灰度图像处理
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
        # 阈值处理
        _,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU )
        # 做一些形态学操作,去一些小物体干扰
        img_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3))
        cv2.erode(img_morph, (3, 3), img_morph, iterations=2)
        cv2.dilate(img_morph, (3, 3), img_morph, iterations=2)
        # 返回值
        return img_morph 
 
       

    # 测量齿数
    def getNumber(self,img):
        """
        """
        # 区域填充
        cnts,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        # print(cnts1)
        #     cvDrawContours
        w,h = img.shape
        draw1 = numpy.zeros((w,h), numpy.uint8)
        draw2 = numpy.zeros((w,h), numpy.uint8)
        cv2.drawContours(draw1,[cnts1],0,255,-1)
        # 求轮廓最小外接圆
        (x,y),r = cv2.minEnclosingCircle(cnts1)
        center = (int(x),int(y))
        radius = int(r)
        draw2 = cv2.circle(draw2,center,radius,255,-1)
        # cv2.imshow('draw1',draw1)
        # cv2.imshow('draw2',draw2)
        # 求draw1与draw2的差值
        xorimg = cv2. subtract(draw2, draw1)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9 ,9 ))
        opening = cv2.morphologyEx(xorimg, cv2.MORPH_OPEN, kernel)
        
        # 求差值的联通区域数量
        cnts,_ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        count = (len(cnts)) 
        # print(count)
        return count


   ################################
    # 检测同步带轮的OD尺寸##########
    ################################
    def getOD(self,img ):
        """
        input : 灰度图像 
        output: OD_No
        """
        contours , hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)#从二值图像thresh中找到轮廓contours和轮廓的层析结构hierarchy
        cntB = sorted(contours,key=cv2.contourArea,reverse=True)[0]
        """
        cntA = sorted(contours,key=cv2.contourArea,reverse=True)[1]
        cntA = contours[1] # 默认内轮廓点集cntA取第二个轮廓
        # 5. 找到重心O(x0,y0),并找到点集B={(x,y)|(x,y)是外轮廓线上的点}
        for i in range(len(contours)): # 遍历轮廓contours
            if hierarchy[0,i,3] == -1: # 当层析结构hierarchy 没有父轮廓时：
                cntB = contours[i]          # 点集B是外轮廓，取当前下标为i的轮廓
                print('B=contours[%s]'%i)
            else:                      # 否则：
                if len(contours[i-1]) < len(contours[i]) :  # 如果当前轮廓比上一个轮廓点数多时：
                    cntA = contours[i]                         # 点集A取当前轮廓
                    # print('A=contours[%s]'%i)

        M = cv2.moments(cntB) # 取点集cntB的矩M
        #print(M)
        cx = int(M['m10']/M['m00']) # 求矩的重心X坐标cx
        cy = int(M['m01']/M['m00']) # 求矩的重心Y坐标cy
        print(cx,cy)


        # 6. 找到点集C={(x,y)|(x,y)是所有内轮廓线点集A所对应的外轮廓点集B上的点，此点距点集A上的点距离最大。}
        A = numpy.matrix(cntA) # 将轮廓cntA转换为矩阵A
        B = numpy.matrix(cntB) # 将轮廓cntB转换为矩阵B
        # print('A = :',A)
        # print('B = :',B)

        from test_demo.getEuclideanDistances_demo import EuclideanDistances as getED # 导入矩阵求欧氏距离的函数。位于./test_demo/getEuclideanDistances_demo 中。
        dC = getED(A,B) # 求量矩阵的欧氏距离，返回结果矩阵dC。
        # print(dC)
        c = numpy.array(dC) # 将矩阵dC转换为numpy.array数组c。
        i = numpy.argmax(c,axis=1) # 求数组每行中的最大值的索引i。
        cntC = numpy.array(B[i])   # 求距离点集cntA最远的点集cntB中的点集cntC。
        cntC = numpy.array(list(set([tuple(t) for t in cntC]))) # 去除点集cntC中的重复值
        print('"*'*10)
        # print(type(cntC))
        # print(cntC)

        # 7. 找到点集D={(x,y)|(x,y)是点集C上所有点距离重心O(x0,y0)的距离的众数±3范围以内的对应的点的集合}
        P = numpy.matrix([[cx,cy]]) # 将重心转换为矩阵P。
        C1 = numpy.matrix(cntC)     # 将点集cntC转换为矩阵C1。
        dP = getED(P,C1)         # 求矩阵P和C1的欧氏距离，返回矩阵dP。
        df = pd.Series   (numpy.array(dP)[0,:]) # 将矩阵dP的数值转换为pandas.Series对象。
        dP_em = df.mode()                    # 求取众数 dP_em
        i = int(len(dP_em)/2)                # 取众数dP_em的中间数据的索引i。PS：众数数量不确定，可以有多个或零个。
        dP_ex = dP_em[i]                     # 取索引为i的众数dP_ex。
        if not dP_ex:                        # 如果众数dP_ex为空：
            dP_ex = numpy.array(dP)[0,:].mean() # 取矩阵dP的平均值。
        d = numpy.array(dP)[0,:]                # 将矩阵dP转换为numpy.array数组d。
        j = numpy.where((d>(dP_ex - 19 )) & (d < (dP_ex+ 19 )))           # 求数组中数值小于众数dP_ex + 6 的索引j。
        cntD = numpy.array(C1[j])               # 求索引为j的数组点集cntD
        cntD = numpy.array(list(set([tuple(t) for t in cntD]))) # 去除点集中的重复元素

        # 8. 用最小二乘法拟合点集D求圆心坐标与直径
        from test_demo.tf_geometry import TfGeometry as Tfg # 导入最小二乘法拟合圆的类。位于: ./test_demo/tf_geometry 中
        od = Tfg(cntD)           # 实例化对象od 输入坐标数据cntD。
        x,y,r = od.fitCircle()   # 计算坐标数据cntD拟合圆的坐标x,y和半径r。
        """
        (x,y),radius = cv2.minEnclosingCircle(cntB)
        print((x,y),radius)

        """
        import pickle

        print(self.var1.get())
        if self.var1.get() == 1:
            BIAODING = r/self.varBiaoDing.get()
            with open('biaoding.csv','wb') as f:
                pickle.dump({"BIAODING":BIAODING},f)
                f.close()

        else:
            with open('biaoding.csv','rb') as f:
                BIAODING = pickle.load(f)["BIAODING"]
                print (BIAODING)
                f.close()

        """

        return x,y, radius
 

    def mainDetect(self):
        # 中值滤波
        median = cv2.medianBlur(self.work_gray,5)
        # 二值化处理
        thresh = self.get_good_thresh_img(median)
        ret,thresh_OD = cv2.threshold(median, 0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)# Otsu's 二值化，将图像middle转换为二值图像thresh
        # 测量OD
        x,y,r= self.getOD(thresh_OD)
        print(x,y,r)
        print(int(x),int(y),int(r))
        # 测量齿数
        count = self.getNumber(thresh)
        print(count)
        # 图像显示
        font = cv2.FONT_HERSHEY_SIMPLEX                                            # 设置opencv的字体
        od_r = round(r*PXCEL/2,3)
        cv2.circle(thresh,(int(x),int(y)),int(r),255,3) # 在二值图片thresh中绘制检测圆
        cv2.circle(thresh,(int(x),int(y)),3,255,3)      # 在二值图片thresh中绘制检测圆的圆心
        thresh=cv2.resize(thresh,None,fx=0.12,fy=0.12,interpolation=cv2.INTER_CUBIC) # 将二值图片缩放至便于观察
        cv2.putText(thresh,str(od_r),(2 , 28),font,1,(255,255,255),2) # 将直径写入二值图片thresh中。
        cv2.putText(thresh,str(count),( 2, 58),font,1,(255,0,0),2) # 将直径写入二值图片thresh中。


        
        # 工作图像

        """
        rows, cols = thresh_OD.shape[:2]
        frame = cv2.resize(thresh_OD, (int(cols / 8), int(rows / 8)))
        cv2.imshow('img',frame)
        """
        
        cv2.namedWindow("检测图")
        cv2.imshow("检测图", thresh)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # 界面处理
        return od_r,count

 
class App():
# class App(tk.Frame):
    def __init__(self,master,endCommand):
        # super().__init__(master)
        self.parent = master
        self.parent.title('同步带轮齿数照合系统 V0.0')
        self.parent.geometry('900x556')
        self.autoDetect = endCommand
        self.autoDetect()
        

        # 设置字体
        ft20 =tkFont.Font(family = 'Fixdsys',size = 20,weight = tkFont.BOLD)
        ft30 =tkFont.Font(family = 'Fixdsys',size = 30,weight = tkFont.BOLD)
        ft40 =tkFont.Font(family = 'Fixdsys',size = 40,weight = tkFont.BOLD)
        ft50 =tkFont.Font(family = 'Fixdsys',size = 50,weight = tkFont.BOLD)
        ft80 =tkFont.Font(family = 'Fixdsys',size = 80,weight = tkFont.BOLD)

 
        # 设置全局变量
        self.varOK = tk.StringVar()
        self.varCeLiangCHiShu = tk.StringVar()
        self.varOD = tk.StringVar()
        self.varLiLunOD = tk.StringVar()

        self.varOK .set('*'*2)
        self.varCeLiangCHiShu .set('0')
        self.varOD.set('0')
        self.varLiLunOD.set('0')

        self.varZhiShiShuID = tk.StringVar()
        self.varXingHao = tk.StringVar()
        self.varLiLunChiShu = tk.StringVar()
        self.varShiJian = tk.StringVar()
        self.varZhuangTai = tk.StringVar()

        self.varZhiShiShuID .set('*'*11)
        self.varXingHao .set('*'*11)
        self.varLiLunChiShu .set('*'*2)
        self.varShiJian .set(time.ctime())
        self.varZhuangTai.set('')



        # 设置菜单栏
        menubar = tk.Menu(self.parent)
        filemenu = tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label='文件',menu=filemenu,font=ft20)
        # 打开 测量记录.csv 文件。 
        filemenu.add_command(label='打开CSV文件',command = lambda : os.popen('测量记录.csv'),font=ft20)
        filemenu.add_separator()##这里就是一条分割线
        filemenu.add_command(label='退出',command = self.parent.quit,font=ft20)
        helpmenu = tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label='帮助',menu=helpmenu)
        helpmenu.add_command(label='帮助',font=ft20,command = lambda : os.popen(r'help.pdf'))
        self.parent.bind("<Button-2>", self.call_back)
        self.parent.config(menu = menubar)

        # 设置frame窗口
        frame = tk.Frame(self.parent)
        frame.pack()

        # 设置标签窗体
        labelFm1= tk.LabelFrame(frame)
        labelFm1.pack(padx=5,pady=5,side=tk.TOP,fill=tk.X)

        labelFm2 = tk.LabelFrame(frame)
        labelFm2.pack(padx=5,pady=5,side=tk.TOP,fill=tk.X)

        labelFm3 = tk.LabelFrame(frame)
        labelFm3.pack(padx=5,pady=5,side=tk.TOP,fill=tk.X)

        labelFm4 = tk.LabelFrame(frame)
        labelFm4.pack(padx=5,pady=5,side=tk.TOP,fill=tk.X)

        # labelFm1
        tk.Label(labelFm1,text='合格判定：',font=ft20,width=10).grid(row=0,column=0)
        self.label0 = tk.Label(labelFm1,textvariable=self.varOK,font=ft50,width=10)
        self.label0.grid(row=0,column=1)
        tk.Label(labelFm1,text='测量时间：',font=ft20,width=10).grid(row=1,column=0)
        self.label1 = tk.Label(labelFm1,textvariable=self.varShiJian,font=ft30,width=30)
        self.label1.grid(row=1,column=1)

        self.label2 = tk.Label(labelFm1,textvariable=self.varZhuangTai  ,font=ft20,width=10)
        self.label2.grid(row=3,column=0)
        """
        self.inputer = tk.Entry(labelFm1,font=ft20,width=40)
        self.inputer.focus()
        self.inputer.grid(row=3,column=1)
        self.inputer.bind('<Return>',self.panDuanShuRu)
        """

        # labelFm2
        tk.Label(labelFm2,text='指示书ID：',font=ft20,width=10).grid(row=1,column=0)
        tk.Label(labelFm2,text='理论齿数：',font=ft20,width=10).grid(row=2,column=0)
        tk.Label(labelFm2,text='理论O.D.：',font=ft20,width=10).grid(row=3,column=0)
        
        self.label4 = tk.Label(labelFm2,textvariable=self.varZhiShiShuID,font=ft30,width=13)
        self.label4.grid(row=1,column=1)
        self.label5 = tk.Label(labelFm2,textvariable=self.varLiLunChiShu,font=ft30,width=13)
        self.label5.grid(row=2,column=1)
        self.label6 = tk.Label(labelFm2,textvariable=self.varLiLunOD,font=ft30,width=13)
        self.label6.grid(row=3,column=1)

        tk.Label(labelFm2,text='产品型号：',font=ft20,width=10).grid(row=1,column=2)
        tk.Label(labelFm2,text='测量齿数：',font=ft20,width=10).grid(row=2,column=2)
        tk.Label(labelFm2,text='  OD尺寸：',font=ft20,width=10).grid(row=3,column=2)

        self.label7 = tk.Label(labelFm2,textvariable=self.varXingHao,font=ft30,width=13)
        self.label7.grid(row=1,column=3)
        self.label8 = tk.Label(labelFm2,textvariable=self.varCeLiangCHiShu,font=ft20,width=13)
        self.label8.grid(row=2,column=3)
        self.label9 = tk.Label(labelFm2,textvariable=self.varOD,font=ft20,width=13)
        self.label9.grid(row=3,column=3)

        self.label1.after(1000,self.trickit)

        # labelFm3
        # button1 = tk.Button(labelFm3,text= "测量" ,command=getOD,font=ft40,width=8)
        button1 = tk.Button(labelFm3,text= "测量" ,command=self.ceLiang,font=ft40,width=7)
        button1.pack(padx=5,pady=5,side = tk.LEFT,expand=True)
        button2 = tk.Button(labelFm3,text= "观察" ,command=self.guancha,font=ft40,width=7)
        button2.pack(padx=5,pady=5,side = tk.LEFT,expand=True)
        # button3 = tk.Button(labelFm3,text= "查询" ,command=self.chaxun,font=ft40,width=7)
        # button3.pack(padx=5,pady=5,side = tk.LEFT,expand=True)
        tk.Button(labelFm3,text='退出',command=self.parent.quit,activeforeground='white',activebackground='red',font=ft40,width=7).pack(padx=5,pady=5,side = tk.LEFT,expand=True)

        # labelFm4
        self.var1 = tk.IntVar()
        checkButton = tk.Checkbutton(labelFm4,text='标定尺寸',variable=self.var1, font=ft20,width =8)
        checkButton .pack(padx=5,pady=5,side = tk.LEFT ,expand=True)
        labelBD = tk.Label(labelFm4,text = '标定尺寸',font=ft20,width=8)
        labelBD .pack(padx=5,pady=5,side = tk.LEFT,expand=True)
        self.varBiaoDing = tk.DoubleVar()
        self.varBiaoDing.set(18.33)
        biaoDing = tk.Entry(labelFm4,textvariable=self.varBiaoDing,font=ft20,width=8)
        biaoDing.pack(padx=5,pady=5,side = tk.LEFT,expand=True)

        # self.parent.update_idletasks()
        # self.parent.update()
        
    def trickit(self):
        self.readMacro()
        self.varShiJian.set(time.ctime())
        self.parent.update()
        self.label1.after(1000,self.trickit)

    # def chaxun(self):
        # pass


    def getimage(self):
        # print the demo information
        #print("")
        #print("-------------------------------------------------------------")
        #print("Sample to show how to acquire mono image continuously and show acquired image.")
        #print("-------------------------------------------------------------")
        #print("")
        #print("Initializing......")
        #print("")

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num is 0:
            print("Number of enumerated devices is 0")
            return

        # open the first device
        cam = device_manager.open_device_by_index(1)

        # if camera is color
        if cam.PixelColorFilter.is_implemented() is True:
            print("This sample does not support color camera.")
            cam.close_device()
            return

        # set continuous acquisition
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # set exposure
        cam.ExposureTime.set(100)

        # set gain
        cam.Gain.set(10.0)

        # start data acquisition
        cam.stream_on()

        # acquire image: num is the image number
        num = 1
        for i in range(num):
            # get raw image
            raw_image = cam.data_stream[0].get_image()
            if raw_image is None:
                print("Getting image failed.")
                continue

            # create numpy array with data from raw image
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                continue

            # show acquired image
            # img = Image.fromarray(numpy_image, 'L')
            # img.show()

            # print height, width, and frame ID of the acquisition image
            print("Frame ID: %d   Height: %d   Width: %d"
                  % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

        # stop data acquisition
        cam.stream_off()

        # close device
        cam.close_device()
        print(numpy_image)

        return numpy_image 



    def ceLiang(self):
        # 原图读入
        image = self.getimage()
        # 检测
        detect = Detect(image) 
        od_r , count = detect.mainDetect()

        # self.label0.config(bg='green')
        self.varCeLiangCHiShu.set(count)
        self.varOD.set(str(od_r))
        od = float(self.varLiLunOD.get())
        if self.varCeLiangCHiShu.get() == self.varLiLunChiShu.get() and abs(od_r-od)<0.3:
            self.varOK.set('OK')
            self.label0.config(bg='green')
            with open('测量记录.csv','a')as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([self.varZhiShiShuID.get(),self.varXingHao.get(),self.varCeLiangCHiShu.get(),self.varShiJian.get()])
        else:
            self.varOK.set('NG')
            self.label0.config(bg='red')
            with open('测量错误记录.csv','a')as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([self.varZhiShiShuID.get(),self.varXingHao.get(),self.varCeLiangCHiShu.get(),self.varShiJian.get()])
                
            
    """
    def panDuanShuRu(self,event=None):
        sp = self.inputer.get().split('@')
        if len(sp) > 1:
            self.varXingHao.set(sp[1])
            m = r'[A-Z]{3,6}(\d\.\d\-)?(\d{2})'
            c = re.search(m,sp[1])
            inputChiShu = int(c.group(2))
            self.varLiLunChiShu.set(inputChiShu)
            self.varZhuangTai.set("*"*2)
            bt = 'MXL|XL|L|H|S2M|S3M|S5M|S8M|P2M|P3M|P5M|P8M|T5|T10|AT5|AT10|GT2|GT3|GT5|YU8'
            x = re.search(bt,sp[1])
            if x:
                inputXingHao = x.group() 
                print(inputXingHao)
            if inputXingHao == 'MXL':
                if inputChiShu == 14:
                    OD =  8.55
                elif inputChiShu == 15:
                    OD =  9.19
                elif inputChiShu == 16:
                    OD =  9.84
                elif inputChiShu == 17:
                    OD = 10.49
                elif inputChiShu == 18:
                    OD = 11.14
                elif inputChiShu == 19:
                    OD = 11.78
                elif inputChiShu == 20:
                    OD = 12.43
                elif inputChiShu == 21:
                    OD = 13.07
                elif inputChiShu == 22:
                    OD = 13.72
                elif inputChiShu == 23:
                    OD = 14.37
                elif inputChiShu == 24:
                    OD = 15.02
                elif inputChiShu == 25:
                    OD = 15.66
                elif inputChiShu == 26:
                    OD = 16.31
                elif inputChiShu == 27:
                    OD = 16.96
                elif inputChiShu == 28:
                    OD = 17.60
                elif inputChiShu == 30:
                    OD = 18.90
                elif inputChiShu == 32:
                    OD = 20.19
                elif inputChiShu == 34:
                    OD = 21.48
                elif inputChiShu == 36:
                    OD = 22.78
                elif inputChiShu == 38:
                    OD = 24.07
                elif inputChiShu == 40:
                    OD = 25.36
                elif inputChiShu == 42:
                    OD = 26.66
                elif inputChiShu == 44:
                    OD = 27.95
                elif inputChiShu == 46:
                    OD = 29.24
                elif inputChiShu == 48:
                    OD = 30.54
                elif inputChiShu == 50:
                    OD = 31.84
                elif inputChiShu == 60:
                    OD = 38.30
                elif inputChiShu == 72:
                    OD = 46.06


            elif inputXingHao =='XL':
                if inputChiShu == 10:
                    OD = 15.66
                elif inputChiShu == 11:
                    OD = 17.28
                elif inputChiShu == 12:
                    OD = 18.90
                elif inputChiShu == 14:
                    OD = 22.13
                elif inputChiShu == 15:
                    OD = 23.75
                elif inputChiShu == 16:
                    OD = 25.36
                elif inputChiShu == 18:
                    OD = 28.60
                elif inputChiShu == 19:
                    OD = 30.22
                elif inputChiShu == 20:
                    OD = 31.83
                elif inputChiShu == 21:
                    OD = 33.45
                elif inputChiShu == 22:
                    OD = 35.07
                elif inputChiShu == 24:
                    OD = 38.30
                elif inputChiShu == 25:
                    OD = 39.92
                elif inputChiShu == 26:
                    OD = 41.53
                elif inputChiShu == 28:
                    OD = 44.77
                elif inputChiShu == 30:
                    OD = 48.00
                elif inputChiShu == 32:
                    OD = 51.24
                elif inputChiShu == 34:
                    OD = 54.47
                elif inputChiShu == 36:
                    OD = 57.70
                elif inputChiShu == 38:
                    OD = 60.94
                elif inputChiShu == 40:
                    OD = 64.17
                elif inputChiShu == 42:
                    OD = 67.14
                elif inputChiShu == 44:
                    OD = 70.64
                elif inputChiShu == 46:
                    OD = 73.87
                elif inputChiShu == 48:
                    OD = 77.11
                elif inputChiShu == 50:
                    OD = 31.84
                elif inputChiShu == 60:
                    OD = 38.30
                elif inputChiShu == 72:
                    OD = 46.06
            elif inputXingHao =='L':
                pass

            elif inputXingHao =='H':
                pass

            elif inputXingHao =='S2M':
                pass

            elif inputXingHao =='S3M':
                pass

            elif inputXingHao =='S5M':
                pass

            elif inputXingHao =='S8M':
                pass

            elif inputXingHao =='P2M':
                pass

            elif inputXingHao =='P3M':
                pass

            elif inputXingHao =='P5M':
                pass

            elif inputXingHao =='P8M':
                pass

            elif inputXingHao =='T5':
                pass

            elif inputXingHao =='T10':
                pass

            elif inputXingHao =='AT5':
                pass

            elif inputXingHao =='AT10':
                pass

            elif inputXingHao =='GT2':
                pass

            elif inputXingHao =='GT3':
                pass

            elif inputXingHao =='GT5':
                pass

            elif inputXingHao =='YU8':
                pass
            self.varLiLunOD.set(OD)

        elif len(self.inputer.get().split('#'))>1:
            self.varZhiShiShuID.set(sp[0].split('#')[0])
        self.inputer.delete(0,tk.END)
        if self.varZhuangTai.get() == "*"*2  :
            self.ceLiang()
            self.varZhuangTai.set("")
    """

    def guancha(self):
        
        while(1):
            frame = self.getimage()

            rows, cols = frame.shape[:2]
            # 工作图像
            frame = cv2.resize(frame , (int(cols / 8), int(rows / 8)))

            # show a frame
            cv2.imshow("capture", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        # cap1.release()
        cv2.destroyAllWindows()


    def call_back(self,event):
        # 按哪个键，在console中打印
        # print("现在的位置是")
        self.ceLiang()

class ThreadRoot:
    #初始化TK root
    def __init__(self, master):
        self.master = master
        self.gui = App(self.master, self.starting)
        self.running=True
 

    # 为方法开第一个单独的线程
    def starting(self):
       #启动线程1
        self.thread1 = threading.Thread(target=self.readMacro)
        self.thread1.setDaemon(True)    #线程守护，即主进程结束后，此线程也结束。否则主进程结束子进程不结束
        self.thread1.start()
 
    def readMacro(self):
        """
        """
        global macroFile 
        global macroCopy
        global timeSave
        while thread1_state:
            # 扫描macro.txt文件改动。
            timeLast = os.path.getmtime(macroFile)
            # 如果发生改动，
            if timeLast > timeSave:
                #print("扫描中……")
                timeSave = timeLast
                os.system(r'copy %s %s'%(macroFile,macroCopy)) 
                f = open(macroCopy,'r')
                fileLines = f.readlines()
                f.close()
                print(fileLines)
                # 读取位置数据
                for item in fileLines:
                    # print(item)
                    patt = r'(\d+)\D+?((-)?\d+\.?\d{0,3})' # 匹配 500 = 34 500=34 500 = 34. 500 =34.5 500 = -34.565 不完全匹配 34.34352345
                    m = re.match(patt,item)
                    if m:
                        shapu = m.group(1)
                        data = m.group(2)
                        # print(shapu,data)
                    #1 输入工件信息、
                        if shapu == '500':
                            self.gui.varLiLunOD.set(data) # O.D
                        elif shapu == '501':
                            self.gui.varLiLunChiShu.set(data) # 齿数
                self.gui.ceLiang()
           # 如果未改动，
            else:
                self.gui.varShiJian.set(time.ctime())
 

thread1_state = True # False #子线程循环标识位（可选项）
"""
cap = cv2.VideoCapture(1)
cap.set(3,2592)
cap.set(4,1944)
"""


root = tk.Tk()
tool = ThreadRoot(root)
root.mainloop()
