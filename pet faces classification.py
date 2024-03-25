import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
import glob
import os

images_fp = './images'

image_names = [os.path.basename(file) for file in glob.glob(os.path.join(images_fp,'*.jpg'))]

image_names

labels = [' '.join(name.split('_')[:-1:]) for name in image_names]

def label_encode(label):
    if label == 'Abyssinian': return 0
    elif label == 'Bengal': return 1
    elif label == 'Birman': return 2
    elif label == 'Bombay': return 3
    elif label == 'British Shorthair': return 4
    elif label == 'Egyptian Mau': return 5
    elif label == 'american bulldog': return 6
    elif label == 'american pit bull terrier': return 7
    elif label == 'basset hound': return 8
    elif label == 'beagle': return 9
    elif label == 'boxer': return 10
    elif label == 'chihuahua': return 11
    elif label == 'english cocker spaniel': return 12
    elif label == 'english setter': return 13
    elif label == 'german shorthaired': return 14
    elif label == 'great pyrenees': return 15

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

features = []
labels = []
IMAGE_SIZE = (224,224)

for name in image_names:
    label = ' '.join(name.split('_')[:-1:])
    label_encoded = label_encode(label)
    if label_encoded != None:
        img = load_img(os.path.join(images_fp, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        image = np.array(img)
        features.append(image)
        labels.append(label_encoded)

features

labels

features_array = np.array(features)
labels_array = np.array(labels)

labels_one_hot = pd.get_dummies(labels_array)

labels_one_hot

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_array, labels_one_hot, test_size = 0.2, random_state = 42)
