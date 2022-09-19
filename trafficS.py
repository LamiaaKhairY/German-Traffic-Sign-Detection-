import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score



data = []
labels = []
classes = 43  # i have 43 of diffrant traffic sign
cur_path = os.getcwd()  # currant work directory
print("current worke directory : " , cur_path)


"""الهدف دلوقتي ان ادخل كل كلاس وادخل علي كل صوره فيه واربطها باللابل علشان اعمل التدريب """
# retrieving the image and their lables
for i in range(classes):
    path = os.path.join(cur_path,"Train",str(i))
    """https://appdividend.com/2022/07/
    /python-os-path-join/#:~:text=The%
    .path.join%20function%20won%E2%80%99t%
    %20if%20a%20component,%E2%80%9Cabsolute%20path%E2%80%9D%20
    and%20everything%20before%20them%20is%20dumped."""
    images = os.listdir(path)
    for a in images:
        try:
              image = Image.open(path+"\\"+a)   #open image
              image = image.resize((30,30))
              image = np.array(image)
              # to use this data to do train 
              data.append(image)  
              labels.append(i)
        except:
            print("i can't read the image")
        
data = np.array(data)
labels = np.array(labels)
print(data.shape , labels.shape)
    

x_train,x_evaluat,y_train,y_evaluat = train_test_split(data,labels,test_size=0.2,random_state=42)
y_train = to_categorical(y_train,43)
y_evaluat = to_categorical(y_evaluat,43)
#https://www.geeksforgeeks.org/python-keras-keras-utils-to_categorical/
print(y_train)

# Building my model
#https://www.educba.com/keras-sequential/
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5), activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32,kernel_size=(5,5), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))  # to avoide overfitting
model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43,activation="softmax"))

#compile
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
epoch = 15
history = model.fit(x_train,y_train,batch_size=32,epochs=epoch,validation_data=(x_evaluat,y_evaluat))
model.save("my_model.h5")
plt.figure(0)
plt.plot(history.history["accuracy"],label="training accuracy")
plt.plot(history.history["val_accuracy"],label="val accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="val loss")
plt.title("loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()
#### test data
y_test = pd.read_csv("Test.csv")
labels = y_test["ClassId"].values
imgs =y_test["Path"].values
data = []


for img in imgs:
    ##img = "trafficS/"+img
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
x_test = np.array(data)
#pred = model.predict_classes(x_test)
predict_x=model.predict(x_test) 
classes_x=np.argmax(predict_x,axis=1)
print("accuracy = ",accuracy_score(labels,classes_x))




