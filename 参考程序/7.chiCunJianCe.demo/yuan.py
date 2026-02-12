import math


def circleLeastFit(points):

    center_x = 0.0;
    center_y = 0.0;
    radius = 0.0;
    if len(points) < 3:
        return false;
     
 
    sum_x = 0.0
    sum_y = 0.0;
    sum_x2 = 0.0
    sum_y2 = 0.0;
    sum_x3 = 0.0
    sum_y3 = 0.0;
    sum_xy = 0.0
    sum_x1y2 = 0.0
    sum_x2y1 = 0.0;

    N = len(points)
    #print(points)
    for x,y in points :
        #x = points[0:1][:];
        #print(x)
        #print(y)
        #y = points[1];
        x2 = x * x;
        y2 = y * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_x3 += x2 * x;
        sum_y3 += y2 * y;
        sum_xy += x * y;
        sum_x1y2 += x * y2;
        sum_x2y1 += x2 * y;
     
 
    C = N * sum_x2 - sum_x * sum_x;
    D = N * sum_xy - sum_x * sum_y;
    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
    G = N * sum_y2 - sum_y * sum_y;
    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
    a = (H * D - E * G) / (C * G - D * D);
    b = (H * C - E * D) / (D * D - G * C);
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;
 
    center_x = a / (-2);
    center_y = b / (-2);
    radius = math.sqrt(a * a + b * b - 4 * c) / 2;
    return center_x,center_y,radius;


points = [[0,2],[2,0],[4,2],[2,4]]

x,y,r = circleLeastFit(points)
print(x,y,r)