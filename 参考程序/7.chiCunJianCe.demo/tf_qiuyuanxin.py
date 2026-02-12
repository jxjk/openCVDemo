import tensorflow as tf
import numpy

# Parameters
learning_rate = 0.1
training_epochs = 3000
display_step = 50

# Training Data, 3 points that form a triangel
train_X = numpy.asarray([0,0.61,1.22,1.83,2.44,3.06,3.67,4.28,4.89,5.5,6.12,6.73])
train_Y = numpy.asarray([0.15,0.88,0.36,1.31,1.48,0.64,1.13,1.49,1.13,1.33,1.35,1.83])

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        if (c - 0) < 0.0000000001:
            break
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            m = sess.run(mean, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "CX=", sess.run(cx), "CY=", sess.run(cy), "Mean=", "{:.9f}".format(m))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "CX=", round(sess.run(cx),2), "CY=", 
        round(sess.run(cy),2), "R=", round(m,2), '\n')