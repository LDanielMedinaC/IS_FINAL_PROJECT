import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image, ImageOps
import numpy as np 
from sklearn.cluster import DBSCAN, KMeans  
import matplotlib.pyplot as plt
from collections import Counter

#create the model
model = tf.keras.Sequential()

#2. define the architecture 
input_layer = tf.keras.layers.Flatten(input_shape=(28,28))
model.add(input_layer)

hidden_layer = tf.keras.layers.Dense(100, activation="sigmoid")
model.add(hidden_layer)

output_layer = tf.keras.layers.Dense(10, activation="sigmoid")
model.add(output_layer)

#3. train the model 
mnist = tf.keras.datasets.mnist
model.compile(optimizer=tf.optimizers.SGD(0.01), loss='sparse_categorical_crossentropy', metrics=['acc'])
(x_train,y_train), (x_test,y_test) = mnist.load_data()
print(type(x_train[1]))
history = model.fit(x_train, y_train, batch_size=300, epochs=20, validation_data=(x_test, y_test))

#4.- Test model
# data = []
# for i in range(10):
#     image = Image.open(("image_{}.png").format(i))
#     gray_image = ImageOps.grayscale(image)
#     pixels = np.asanyarray(gray_image)
#     data.append(pixels)
# labels = [1,2,3,4,5,6,7,8,9,0]
# print(type(data[0]))
# predictions = model.predict(data[0])
# plt.figure(figsize=(10, 10))
# i, j = 0, 0
# for p in predictions:
#     if(p.argmax() != int(labels[i]) and j < 25):
#         plt.subplot(5, 5, j+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(x_train[i], cmap='gray')
#         plt.xlabel("{}/{}".format(y_test[i], p.argmax()))
#         j += 1
#     i += 1
# plt.show()

for i in range(10):
    image = Image.open(("image_{}.png").format(i)).convert("L")
    img_arr = np.array(image)
    img_arr = img_arr.reshape(1,28,28,1)
    pred = model.predict_classes(img_arr)
    print(pred)


#5. Display the results
# plt.figure(figsize=(10,6))
# plt.subplot(2,2,1)
# plt.plot(range(len(history.history['acc'])), history.history['acc'])
# plt.ylabel('accuracy')
# plt.xlabel('epochs')
# plt.subplot(2,2,2)
# plt.plot(range(len(history.history['loss'])), history.history['loss'])
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.show()

