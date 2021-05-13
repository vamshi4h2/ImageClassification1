import cv2
import keras
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
CATEGORIES = ['Cat', 'Dog']
import os


def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60,60 ))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr
print("enter the image name with the extension:",end="")
x=input()
IMAGE=os.path.join(r'C:\Users\vamshi\PycharmProjects\ImageRecognition\Test',x)
model = keras.models.load_model(r'C:\Users\vamshi\PycharmProjects\ImageRecognition\trained.model')
prediction = model.predict([image(IMAGE)])


img=mpimg.imread(IMAGE)
#5imgplot=plt.imshow(img)
plt.show()



img_array=cv2.imread(IMAGE,cv2.IMREAD_GRAYSCALE)
new_array=cv2.resize(img_array,(60,60))
#plt.imshow(new_array,cmap="gray")
plt.show()




img=mpimg.imread(IMAGE)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.axis('off')
plt.show()

print(CATEGORIES[int(prediction[0][0])])