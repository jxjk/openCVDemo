# import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt 
import math


class TfGeometry:
	def __init__(self,points1 = None,points2=None):
		self.points1 = np.asarray(points1)
		self.x = self.points1[:,0]
		self.y = self.points1[:,1]
		self.points2 = points2
	

	def getLine(self):
		# Parameters
		learning_rate = 0.00009
		training_epochs = 3000000
		display_step = 50

		# create data
		x_data = self.x  
		y_data = self.y

		  
		Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))  
		biases = tf.Variable(tf.zeros([1]))
 	  
		y = Weights*x_data + biases 

		  
		loss = tf.reduce_mean(tf.square(y-y_data))  
		  
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate)   
		train = optimizer.minimize(loss)  
		 
		init = tf.global_variables_initializer()  
		  
		with tf.Session() as sess: 
			sess.run(init)
			for epoch in range(training_epochs):
				sess.run(train)

				c = sess.run(loss)
				if (c - 0) < 0.0001: #6
					break

				if epoch % display_step == 0:  
					print(epoch,sess.run(Weights),sess.run(biases))

			W = sess.run(Weights)
			b = sess.run(biases)
			return round(W[0],2),round(b[0],2)


	def getCircle(self):
		# Parameters
		learning_rate = 0.1
		training_epochs = 300000
		display_step = 2

		# Training Data, 3 points that form a triangel
		train_X = self.x#np.asarray([0,5,10])#
		train_Y = self.y#np.asarray([5,0,5])#

		# tf Graph Input
		X = tf.placeholder("float")
		Y = tf.placeholder("float")

		# Set vaibale for center
		cx = tf.Variable(3, name="cx",dtype=tf.float32)
		cy = tf.Variable(3, name="cy",dtype=tf.float32)

		# Caculate the distance to the center and make them as equal as possible
		distance = tf.pow(tf.add(tf.pow((X-cx),2),tf.pow((Y-cy),2)),0.5)
		mean = tf.reduce_mean(distance)
		cost = tf.reduce_sum(tf.pow((distance-mean),2)/3)
		# Gradient descent
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		# Start training
		with tf.Session() as sess:
			sess.run(init)

			# Fit all training data
			for epoch in range(training_epochs):
				sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
				c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
				if (c - 0) < 20:
					break
				#Display logs per epoch step
				if (epoch+1) % display_step == 0:
					c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
					m = sess.run(mean, feed_dict={X: train_X, Y:train_Y})
					print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
						"CX=", sess.run(cx), "CY=", sess.run(cy), "Mean=", "{:.9f}".format(m))

			#print ("Optimization Finished!")
			training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
			#print ("Training cost=", training_cost, "CX=", round(sess.run(cx),2), "CY=", 
				#round(sess.run(cy),2), "R=", round(m,2), '\n')
			return round(sess.run(cx),2), round(sess.run(cy),2), round(m,2)


	def getP2L(self):
		#w,b=self.getLine()
		w,b = self.Least_squares()
		print(w,b)
		distance_sum = 0
		for px,py in self.points2[:]:
			distance = (px*w-1*py+b)/((w**2+1)**0.5)
			distance_sum += distance
			#print(distance_sum)
			#print(len(self.points2))
		return (distance_sum/len(self.points2))


	def circleLeastFit(self):

		center_x = 0.0;
		center_y = 0.0;
		radius = 0.0;
		if len(self.points1) < 3:
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

		N = len(self.points1)
		#print(points)
		for x,y in self.points1 :
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

	def Least_squares(self):
		x = self.x
		y = self.y
		x_ = x.mean()
		y_ = y.mean()
		m = np.zeros(1)
		n = np.zeros(1)
		k = np.zeros(1)
		p = np.zeros(1)
		for i in np.arange(len(x)):
			k = (x[i]-x_)* (y[i]-y_)
			m += k
			p = np.square( x[i]-x_ )
			n = n + p
		a = m/n
		b = y_ - a* x_
		return a,b


#"""
if __name__ == "__main__":
	"""
	a = TfGeometry(points1= [[570, 649], [571, 650], [572, 650], [573, 651], [574, 651], [602, 663], [603, 663], [604, 664]])
	aw,ab = a.getLine()
	print('aw=%s ab=%s'%(aw,ab))

	b = TfGeometry(points1= [[2,0],[1,1],[0,2]])
	bw,bb = b.getLine()
	print('bw=%s bb=%s'%(bw,bb))

	c = TfGeometry(points1= [[0,5],[5,0],[5,10]])
	print(c.getCircle())
	"""
	d = TfGeometry(points1= [[0,0],[1,1],[2,2]],points2=[[0.0,1.0],[1,2],[2,3]])
	print(d.getP2L())
#"""[[570, 649], [571, 650], [572, 650], [573, 651], [574, 651], [602, 663], [603, 663], [604, 664], [605, 664], [606, 665], [634, 676], [635, 677], [636, 677], [637, 678], [638, 678], [666, 690], [667, 690], [668, 691], [669, 691], [698, 704], [699, 704], [730, 717]] [[743, 286], [775, 299], [806, 312], [807, 312], [838, 325], [839, 325], [869, 338], [870, 338], [871, 338], [901, 351], [902, 351], [903, 351], [933, 364], [934, 364], [935, 365], [965, 377], [966, 377], [967, 378], [997, 390], [998, 390], [999, 391], [1029, 403], [1030, 403], [1031, 404]]
