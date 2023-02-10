import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
physical_Dedicated = tf.config.list_physical_devices('GPU')
print(tf.test.is_built_with_cuda())
print(tf.version.VERSION)
print(tf.test.gpu_device_name())
tf.config.list_physical_devices()
print(tf.config.list_physical_devices())