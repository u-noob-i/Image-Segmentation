import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
import os

def load_path(train_path, train_mask_path, test_path, test_mask_path):
    train_x = [f for f in os.listdir(train_path)]
    train_y = [f for f in os.listdir(train_mask_path)]
    test_x = [f for f in os.listdir(test_path)]
    test_y = [f for f in os.listdir(test_mask_path)]
    return train_x, train_y, test_x, test_y

def read_image(path, h, w):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img/255.0  
    img = cv2.resize(img, (h, w))
    return img

def read_mask(path, h, w):
    mask = cv2.imread(path)
    mask = cv2.resize(mask, (w, h))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask[:, :, None]
    return mask

def get_data(x_path, y_path, x_ids, y_ids, h, w):
    #getting Train images
    X = np.zeros((len(x_ids), h, w, 3), dtype=np.float32)
    Y = np.zeros((len(y_ids), h, w, 1), dtype=bool)
    #Image
    for n, id_ in tqdm(enumerate(x_ids), total=len(x_ids)):
        path = x_path+'/'+id_
        X[n] = read_image(path, h, w)
    #Mask
    for n, id_ in tqdm(enumerate(y_ids), total=len(y_ids)):
        path = y_path+'/'+id_
        Y[n] = read_mask(path, h, w)
    return X, Y

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y