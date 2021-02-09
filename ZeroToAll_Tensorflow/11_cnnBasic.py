from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

image = tf.constant([[[[1],[2],[3]],
                   [[4],[5],[6]], 
                   [[7],[8],[9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.numpy().reshape(3,3), cmap='Greys') #(1,3,3,1) batch, h, w, channel
# plt.show()
"""
1 filter (2,2,1,1) with padding: VALID  2x2 size,  channel1, filter num1
"""
# print("image.shape", image.shape)
# weight = np.array([[[[1.]],[[1.]]],
#                    [[[1.]],[[1.]]]])
# print("weight.shape", weight.shape)
# weight_init = tf.constant_initializer(weight)
# conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', 
#                              kernel_initializer=weight_init)(image)#맨마지막이 입력값
# print("conv2d.shape", conv2d.shape)
# print(conv2d.numpy().reshape(2,2))
# plt.imshow(conv2d.numpy().reshape(2,2), cmap='gray')
# plt.show()
"""
3 filters (2,2,1,3) with padding: VALID  2x2 size,  channel1, filter num 3
첫필터  2필터    3필터
1 1    10 10    -1 -1
1 1    10 10    -1 -1
"""
weight = np.array([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                   [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(feature_map.reshape(3,3), cmap='gray')
plt.show()
