import tensorflow as tf
import numpy as np
import cv2
import os
from glob import glob

h = 256
w = 256

def load_path(x, y):
    x = sorted(glob(os.path.join(x, "*")))
    y = sorted(glob(os.path.join(y, "*")))
    return x, y

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (h, w))
    img = img/255.0
    img = img.astype(np.float32)
    return img

def read_mask(path):
    mask = cv2.imread(path)
    mask = cv2.resize(mask, (h, w))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask / 255.0 
    mask = mask[:, :, None]
    mask = mask.astype(np.bool)
    return mask

def get_data(x_path, y_path):
    def get_each(x, y):
        x = read_image(x.decode())
        y = read_mask(y.decode())
        return x, y

    x, y = tf.numpy_function(get_each, [x_path, y_path], [tf.float32, tf.bool])
    x.set_shape([h, w, 3])
    y.set_shape([h, w, 1])
    return x, y

def get_classification_data(x, y):
    def get_image(x, y):
        x = read_image(x.decode())
        y = y.astype(np.float32)
        return x, y
    x, y = tf.numpy_function(get_image, [x, y], [tf.float32, tf.float32])
    x.set_shape([h, w, 3])
    y.set_shape([1])
    return x, y

def tf_data(x, y, buffer_size = 100, batch_size = 4, num_epochs = 1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(get_data)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(4)
    return dataset

def tf_classification(x, y, buffer_size = 100, batch_size = 4, num_epochs = 1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(get_classification_data)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(4)
    return dataset

def get_np_images(X_path, Y_path):
    X = np.zeros((len(X_path), h, w, 3), dtype=np.float32)
    Y = np.zeros((len(Y_path), h, w, 1), dtype=np.bool)
    for idx, x in enumerate(X_path):
        img = read_image(x)
        X[idx] = img
    for idx, y in enumerate(Y_path):
        mask = read_mask(y)
        Y[idx] = mask
    return X, Y