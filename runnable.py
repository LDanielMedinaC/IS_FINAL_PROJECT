from nn_class import createNN
from read_image import separateImge
from classify_imgs import tag_image

#createNN()
# separateImge("car_plate.png", "img_plate")
# separateImge("handwrittenPhone.png", "hw_image")
tag_image(13,"hw_image", [-1,3,-1,3,7,8,8,7,6,5,2,0,-1])
tag_image(6,"img_plate", [-1,3,9,5,7,8])
