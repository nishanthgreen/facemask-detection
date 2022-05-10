import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.python.keras.models import sequential
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()

model.add(Conv2D(400,(3,3),input_shape=data.shape[1:]))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3),input_shape=data.shape[1:]))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(50,activation='elu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.35,random_state=3)

history = model.fit(train_data,train_target,epochs=25,verbose=1)

print(model.evaluate(test_data,test_target))

model.save('face_150.hdf5')

print(history)