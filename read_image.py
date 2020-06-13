import PIL
from PIL import Image, ImageOps
import numpy as np 
from sklearn.cluster import DBSCAN, KMeans  
import matplotlib.pyplot as plt
from collections import Counter

MIN_X, MIN_Y, MAX_X, MAX_Y = 0,1,2,3

def getPixels(path):
    #1,INPUT: Read the image
    image = Image.open(path)

    #2. convert to gray scale
    gray_image = ImageOps.grayscale(image)
    
    #3. Separeta the umage unto digits (Clustering)
    #3.1 Convert the pixels into a list of points
    #print(gray_image)
    pixels = np.asanyarray(gray_image)
    return pixels

def getPoints(pixels):
    points = []
    maxX = len(pixels)
    maxY = len(pixels[0])


    for i in range(maxX):
        for j in range(maxY):
            if pixels[i][j] < 100:
                points.append((i,j))

    return points

def getClusters(points):
    #3.2 finding cluster of pixels to set numbers

    clustering = DBSCAN(eps=3, min_samples=5)
    clusters = clustering.fit_predict(points)

    points_2 = np.array(points)
    plt.scatter(points_2[:, 1], points_2[:, 0], c=clusters, cmap="plasma")
    plt.show()

    n_cluster = len(Counter(clusters).keys())

    imgs = [[] for _ in range(n_cluster)]# Create 1 array per cluster
    for i in range(len(clusters)):
        imgs[clusters[i]].append(points[i])

    return imgs

def getBounds(imgs):
    bounds = [[100000,100000,0,0] for _ in range(len(imgs))]
    #Find boundaries for every image
    i=0
    for img in imgs:
        for x,y in img:
            if x < bounds[i][MIN_X]: bounds[i][MIN_X] = x
            if x > bounds[i][MAX_X]: bounds[i][MAX_X] = x
            if y < bounds[i][MIN_Y]: bounds[i][MIN_Y] = y
            if y > bounds[i][MAX_Y]: bounds[i][MAX_Y] = y
        i += 1
    return bounds

def arrToImage_arr(img, b):
    size = max(b[MAX_X] - b[MIN_X], b[MAX_Y]-b[MIN_Y]) + 7
    middle_x = (size-b[MAX_X] + b[MIN_X])//2
    middle_y = (size-b[MAX_Y] + b[MIN_Y])//2
    new_img = [[0 for _ in range(size)] for _ in range(size)]
    for p in img:
        x,y = p
        new_img[x-b[MIN_X]+middle_x][y-b[MIN_Y]+middle_y] = 255
    return new_img

def scaleImage(imgs, name_img):
    bounds = getBounds(imgs)
    for i in range(len(imgs)):
        new_img = arrToImage_arr(imgs[i], bounds[i])
        img = Image.fromarray(np.uint8(new_img))
        basewidth = 28
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        # x,y = img.size
        # if(y == 27):
        #     img_arr = np.asanyarray(img)
        #     arr = [255 for _ in range(28)]
        #     np.append(img_arr, arr)
        #     img = Image.fromarray(np.uint8(img_arr))
        #     print("->",img.size, len(img_arr), len(img_arr))
        print(img.size)
        img.save((name_img+"{}.png".format(i)))
def separateImge(path, name_img):
    pixels = getPixels(path)
    points = getPoints(pixels)
    imgs = getClusters(points)
    scaleImage(imgs,name_img)


separateImge("car_plate.png", "img_plate")
separateImge("handwrittenPhone.png", "hw_image")






