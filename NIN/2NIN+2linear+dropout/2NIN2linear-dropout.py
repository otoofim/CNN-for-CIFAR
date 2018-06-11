
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
from sklearn.metrics import confusion_matrix
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider



# In[2]:

train_data = CIFAR10DataProvider('train', batch_size=128)
valid_data = CIFAR10DataProvider('valid', batch_size=128)


# In[3]:

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[4]:

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,
                   use_pooling_ave=False,
                   use_pooling_ave2=False):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    if use_pooling_ave:
        layer = tf.nn.avg_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    if use_pooling_ave2:
        layer = tf.nn.avg_pool(value=layer,
                               ksize=[1, 8, 8, 1],
                               strides=[1, 8, 8, 1],
                               padding='SAME')
    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# In[5]:

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    print(layer_flat)
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# In[6]:

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[7]:

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 72         # There are 16 of these filters.

# Fully-connected layer.
fc_size = 128           


# In[8]:

# We know that CIFAR-10 images are 32 pixels in each dimension.
img_size = 32

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 3 channels.
num_channels = 3

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size * num_channels


# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[9]:

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# In[10]:

layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=5,
                   num_filters=72,
                   use_pooling=False)


# In[11]:

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=72,
                   filter_size=1,
                   num_filters=32,
                   use_pooling=False,
                   use_pooling_ave=False)


# In[12]:

layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=32,
                   filter_size=1,
                   num_filters=32,
                   use_pooling=True)
layer_conv3 = tf.nn.dropout(layer_conv3, keep_prob=0.5)


# In[13]:

layer_conv4, weights_conv4 =     new_conv_layer(input=layer_conv3,
                   num_input_channels=32,
                   filter_size=3,
                   num_filters=72,
                   use_pooling=False)


# In[14]:

layer_conv5, weights_conv5 =     new_conv_layer(input=layer_conv4,
                   num_input_channels=72,
                   filter_size=1,
                   num_filters=32,
                   use_pooling=False,
                   use_pooling_ave=False)


# In[15]:

layer_conv6, weights_conv6 =     new_conv_layer(input=layer_conv5,
                   num_input_channels=32,
                   filter_size=1,
                   num_filters=32,
                   use_pooling=False,
                   use_pooling_ave2=False)


# In[16]:

layer_flat, num_features = flatten_layer(layer_conv6)


# In[17]:

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob=0.6)


# In[18]:

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


# In[19]:

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[20]:

epochs=50
init = tf.global_variables_initializer()
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: input_batch, y_true: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
        with open("sep1_1_1_{0}.txt".format(num_filters1), "a") as myfile:
            myfile.write('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}\n'.format(e + 1, running_error, running_accuracy))
        if (e + 1) % 5 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [cost, accuracy], 
                    feed_dict={x: input_batch, y_true: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
            with open("sep1_1_1_{0}.txt".format(num_filters1), "a") as myfile:
                myfile.write('                 err(valid)={0:.2f} acc(valid)={1:.2f}\n'.format(valid_error, valid_accuracy))
            
        


# In[ ]:



