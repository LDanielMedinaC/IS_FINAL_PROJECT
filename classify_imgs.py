import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image, ImageOps
import numpy as np 
from sklearn.cluster import DBSCAN, KMeans  
import matplotlib.pyplot as plt
from collections import Counter

model = tf.keras.models.load_model('trained_model')



def tag_image(n_imgs, name_img, num_tag):
    images = []
    for i in range(n_imgs):
        image = Image.open(name_img+("{}.png").format(i)).convert("L")
        img_arr = np.array(image)
        img_arr = img_arr.reshape(1,28,28,1)
        images.append(img_arr)
        pred = model.predict_classes(img_arr)
        plt.subplot(3, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        plt.xlabel("{}/{}".format(pred, num_tag[i]))
        print(pred)
    plt.show()

