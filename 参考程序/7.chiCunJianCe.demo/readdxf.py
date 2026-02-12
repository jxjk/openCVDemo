from Point import Point 
import re


class DXFReaderImpl:  
    def __init__(self,file):  
        self.file = file  
        self.points = [] ## 用于记录点实体的坐标值  
        self.points_line = []  ## 用于记录线段的各端点坐标值，包括直线和折线两种线型  
        self.points_polygon = [] ## 用于纪录多边形的顶点坐标值 
        self.points_Circel = [] ##用于记录圆的x,y,z,r数值 
  
    def readDXF(self):  
        firstLine =""  
        secondLine = ""  
        secondLine = self.file.readline().strip()  
  
        while secondLine != "EOF":  
            if firstLine.strip() == "0" and secondLine.strip()== "LWPOLYLINE":  
                self.readPolyline()  
  
            if firstLine.strip() == "0" and secondLine.strip() == "LINE":  
                self.readLines()  
  
            if firstLine.strip() == "0" and secondLine.strip() == "ARC":  
                self.readArc()

            if firstLine.strip() == "0" and secondLine.strip() == "CIRCLE":  
                self.readCircle()


            if firstLine.strip() == "0" and secondLine.strip() == "POINT":  
                self.readPoint()

            firstLine = secondLine  
            secondLine = self.file.readline().strip()  
  
        #print ("there are " + str(i) + "polyline")  
  
##  
##    def readPolygon(self):  
##        pass  
## read polyline. In dxf file,polygon is a closed polyline  
    def readPolyline(self):  
        counter = 0  
        numofvertex = 1  
        flagofPolygon = 0  
        x = 0  
        y = 0  
        firstLine = "LWPOLYLINE"  
        secondLine = self.file.readline().strip()  
        pointList = []  
        while counter<=numofvertex:  
            if firstLine == "90":  
                numofvertex = int(secondLine)  
            if firstLine == "70":  
                flagofPolygon = int(secondLine)  
            if firstLine == "10":  
                x = float(secondLine)  
            if firstLine == "20":  
                y = float(secondLine)  
                pointList.append(Point(x,y))  
                counter = counter + 1  
            firstLine = secondLine  
            secondLine = self.file.readline().strip()  
  
        if flagofPolygon == 0:  
            self.points_line.append(pointList)  
        else:  
            self.points_polygon.append(pointList)  
  
    def readPoint(self):  
        firstLine = "POINT"  
        secondLine = self.file.readline().strip()  
        x = 0.0  
        y = 0.0  
        while firstLine != "30":  
            if firstLine == "10":  
                x = float(secondLine)  
            if firstLine == "20":  
                y = float(secondLine)  
        firstLine = secondLine  
        secondList = self.file.readline().strip();  
        self.points.append(Point(x,y))  
## read  straight line  
    def readLines(self):  
        x1 = 0.0  
        y1 = 0.0  
        x2 = 0.0  
        y2 = 0.0  
        firstLine = "AcDbLine"#"POINT"  
##        secondLine = ""  
        secondLine = self.file.readline().strip()
        #i = 0  
        while firstLine != "31":  
            #i += 1
            #print(i)
            #print(secondLine)
            if firstLine == "10":  
                x1 = float(secondLine)  
            if firstLine == "20":  
                y1 = float(secondLine)  
            if firstLine == "11":  
                x2 = float(secondLine)  
            if firstLine == "21":  
                y2 = float(secondLine)  
            firstLine = secondLine  
            secondLine = self.file.readline().strip()  
  
        tempLine = []  
        tempLine.append(Point(x1,y1))  
        tempLine.append(Point(x2,y2))  
        self.points_line.append(tempLine)
        print(x1,y1,x2,y2) 


    def readArc(self):
        pass


    def readCircle(self):
        pass
        x1 = 0.0  
        y1 = 0.0 
        z1 = 0.0 
        r = 0.0  
        firstLine = "AcDbCircle"#"POINT"  
##        secondLine = ""  
        secondLine = self.file.readline().strip()
        while (firstLine != "40"):  
            if firstLine == "10": 
                x1 = float(secondLine)  
            if firstLine == "20":  
                y1 = float(secondLine)  
            if firstLine == "30":  
                z1 = float(secondLine)  
            firstLine = secondLine  
            secondLine = self.file.readline().strip()
        if firstLine == "40":  
            r = float(secondLine)    
  
        tempLine = []  
        tempLine.append([x1,y1,z1,r])  
        #tempLine.append(Point(x2,y2))  
        self.points_Circel.append(tempLine)
        print(x1,y1,z1,r) 
  
  
if __name__=="__main__":  
    file = open("./test.dxf","r")  
  
    reader = DXFReaderImpl(file)  
    reader.readDXF() 
    lp = reader.points_Circel 
    print(lp)
    i = 1  
    for temp in reader.points_polygon:  
        print (" this is the " + str(i) + " polygon" ) 
        for points in temp:  
            print (str(points.x) + "   " + str(points.y) ) 
        i = i + 1  
##    i = 1  
##    for temp in reader.points_line:  
##        print " this is the " + str(i) + " polyline"  
##        for points in temp:  
##            print str(points.x) + "  " + str(points.y)  
##        i += 1  
    print ("over")
