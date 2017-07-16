import tensorflow as tf 
import numpy as np

# cell = tf.nn.rnn_cell.BasicLSTMCell(2048, state_is_tuple = True)
cell = tf.nn.rnn_cell.BasicLSTMCell(2048, state_is_tuple = False)
init_state_c = tf.placeholder(tf.float32, [None, 2048])
init_state_h = tf.placeholder(tf.float32, [None, 2048])
#init_state = tf.placeholder(tf.float32, [None, 2048 * 2])
x = tf.placeholder(tf.float32, [None, 200])
output, (state_c, state_h) = cell(x, (init_state_c, init_state_h))
#output, state = cell(x, init_state)
print(type(output), output.shape)
print(type(state_c), state_c.shape)
print(type(state_h), state_h.shape)
init_op = tf.global_variables_initializer()
print("finished !")

x_ = np.reshape(np.random.uniform(-1, 1, 200), [1, 200])
c_ = np.reshape(np.random.uniform(-1, 1, 2048), [1, 2048])
h_ = np.reshape(np.random.uniform(-1, 1, 2048), [1, 2048])

with tf.Session() as sess:
  sess.run(init_op)
  out, sttc, stth = sess.run((output, state_c, state_h), feed_dict = {x: x_, init_state_c: c_, init_state_h: h_})
  print(out)
  print(sttc)
  print(stth)

