import cv2
import os
import numpy as np
from tensorflow.python.keras.utils import np_utils

dataset = "dataset"
categories = os.listdir(dataset)
label = [0,1]
label_dict = dict(zip(categories,label))

print(label_dict)

imgsize =100
data =[]
target =[]

for category in categories:
    folder_path = os.path.join(dataset,category)
    img_names = os.listdir(folder_path)

    for img in img_names:
        img_path = os.path.join(folder_path,img)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray,(imgsize,imgsize))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('exception:',e)

data = np.array(data)/255.0
print(data.shape)
data = np.reshape(data,(data.shape[0],imgsize,imgsize,1))
print(data.shape)

target = np.array(target)
print(target)
new_target = np_utils.to_categorical(target)
print(new_target)
np.save('data',data)
np.save('target',new_target)

