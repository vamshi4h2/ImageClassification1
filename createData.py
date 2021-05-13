import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

Datadir=r"C:\Users\vamshi\PycharmProjects\ImageClassification\PetImages/"

CATAGORIES=["cat","dog"]
#r"C:\Users\vamshi\PycharmProjects\ImageClassification\PetImages/"
for i in CATAGORIES:
    path=os.path.join(Datadir,i)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
img_size=60
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap="gray")
plt.show()


training_data=[]

def create_training_data():
    for i in CATAGORIES:
        path=os.path.join(Datadir,i)
        class_num=CATAGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)

X=[]
y=[]

for features,lable in training_data:
    X.append(features)
    y.append(lable)


#print(X[0].reshape(-1, img_size, img_size, 1))

X=np.array(X).reshape(-1,img_size,img_size,1)

import pickle

pickle_out=open(r"C:\Users\vamshi\PycharmProjects\ImageClassification\X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open(r"C:\Users\vamshi\PycharmProjects\ImageClassification\y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()