

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import random


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(input_tensor, num_in, num_out, kernel_size=[3, 3]):
    weights = weight_variable([kernel_size[0], kernel_size[1], num_in, num_out])
    bias = bias_variable([num_out])
    output_tensor = tf.nn.leaky_relu(tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
    num_in = num_out
    return output_tensor, num_in


def max_pool(x, pool_count):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), 2*num_out, pool_count * 2


def dense_layer(c_in, num_in, num_out, p_count):
    num_neurons = x_len // p_count * y_len // p_count * num_in
    print(num_neurons)
    W_fc1 = weight_variable([num_neurons, num_out])
    b_fc1 = bias_variable([num_out])
    h_pool2_flat = tf.reshape(c_in, [-1, num_neurons])
    h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return tf.nn.dropout(h_fc1, keep_prob), num_out


def finalLayer(x, num_in, num_out):
    W_fc1 = weight_variable([num_in, num_out])
    b_fc1 = bias_variable([num_out])
    output_neuron = tf.nn.leaky_relu(tf.matmul(x, W_fc1) + b_fc1)
    return output_neuron


def import_data(file_path, num_classes):
    loaded = np.load(file_path)
    data = loaded['images']
    labels = loaded['counts']
    # Convert the int numpy array into a one-hot matrix.
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels = (np.arange(num_classes) == labels_np[:, None]).astype(np.float32)
    return data, labels


def create_data(amount, x_len):
    sig_to_noise = 4
    num_points_range = [i + 3 for i in range(3)]
    n = x_len
    radius_range = [2, 3]
    images = []
    counts = []
    for j in range(amount):
        image = np.random.rand(x_len, x_len)
        for i in range(random.choice(num_points_range)):
            a, b = np.random.randint(2, x_len - 2), np.random.randint(2, x_len - 2)
            r = random.choice(radius_range)
            y, x = np.ogrid[-a:n - a, -b:n - b]
            mask = x * x + y * y <= r * r
            array = np.ones((n, n)) * sig_to_noise
            image += mask * array
        counts.append(i + 1)
        images.append(image)
    labels_np = np.array(counts).astype(dtype=np.uint8)
    labels = (np.arange(num_classes) == labels_np[:, None]).astype(np.float32)
    return images, labels



# fileloc = '/media/parthasarathy/Stephen Dedalus/generic_blob_images/images-counts.npz'
x_len = 128
y_len = 128
num_classes = 3
# data, labels = import_data(fileloc, num_classes)
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels)


num_epochs = 1000
batch_size = 100
starter_learning_rate = 0.0001
decay_rate = 0.97
kernels_in_first_layer = 16
dense_layer_neuron_num = 1024


sess = tf.InteractiveSession()


# placeholder for our data input and output
data_size = x_len * y_len
p_count = 1
num_in = 1
num_out = kernels_in_first_layer
x = tf.placeholder(tf.float32, shape=[None, data_size])
x_image = tf.reshape(x, [-1, x_len, y_len, 1])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
# Get the shape of the training data.

#first layer
c1, num_in = conv_layer(x_image, num_in, num_out)
p1, num_out, p_count = max_pool(c1, p_count)
# second layer
c2, num_in = conv_layer(p1, num_in, num_out)
p2, num_out, p_count = max_pool(c2, p_count)
#first layer
c3, num_in = conv_layer(p2, num_in, num_out)
p3, num_out, p_count = max_pool(c3, p_count)
# second layer
c4, num_in = conv_layer(p3, num_in, num_out)
p4, num_out, p_count = max_pool(c4, p_count)



# dense layer
dense = dense_layer(p4, num_in, dense_layer_neuron_num, p_count)
output_neuron = finalLayer(dense, num_in=int(dense_layer_neuron_num), num_out=1)
#   loss - optimizer - evaluation
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           120, decay_rate, staircase=True)
loss = tf.losses.mean_squared_error(y_, output_neuron)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

sess.run(tf.global_variables_initializer())

print(str(num_epochs) + ' epochs')
ac_list = []
for epoch in range(num_epochs):
    batch_data, batch_labels = create_data(batch_size, x_len)
    batch_data = [np.array(i).flatten() for i in batch_data]
    optimizer.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
    if epoch % 50 == 0:
            train_loss = loss.eval(feed_dict={
                x: batch_data, y_: batch_labels})
            print("training loss %g" % train_loss)
            print('learning rate = ' + str(learning_rate.eval()))
            print('output neuron = ' + str(output_neuron.eval(feed_dict={
                x: batch_data})[0]))
            print('global step = ' + str(global_step.eval()))
    ac_list.append(train_loss)
plt.plot(ac_list)


#  Test accuracy
test_data, test_labels = create_data(100, x_len)
test_data = [np.array(i).flatten() for i in test_data]
print("test set accuracy %g" % accuracy.eval(feed_dict={
    x: test_data, y_: test_labels, keep_prob: 1.0}))


