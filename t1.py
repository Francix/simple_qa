import tensorflow as tf

x1 = tf.placeholder(tf.float32, [None, 1])
x2 = tf.placeholder(tf.float32, [None, 1])
x3 = tf.placeholder(tf.float32, [None, 1])
x = tf.stack([x1, x2, x3])
print(x.shape)
x = tf.squeeze(x)
print(x.shape)
x = tf.reshape(x, [3, -1, 1])
print(x.shape)
