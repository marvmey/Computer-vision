

import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import listdir
from os.path import join
import cv2
import os
import random
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow import keras
import os
import sys
from PIL import Image
sys.modules['Image'] = Image
from array import array
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import load_model

# CHOOSE OPTIMIZER YOU WANT
# rms = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# adam = Adam(learning_rate=0.0001, decay=1e-6)
# SGD = SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")

#  VARIABLES
epochs =
dir_dataset = ''
batch_size =
target_size = (,)
input_shape= (,,)
shuffle = False
optimizers =
name_of_model =''
num_class =
class_mode =
color_mode =
SEED =

#  PREPARE DATASET

label = []
path = []


for dirname, _,filenames in os.walk(dir_dataset):
    for filename in filenames:
        if os.path.splitext(filename)[1]=='.png':
            if dirname.split()[-1]!='GT':
                label.append(os.path.split(dirname)[1])
                path.append(os.path.join(dirname,filename))

df = pd.DataFrame(columns=['path','label'])
df['path']=path
df['label']=label
df['label']=df['label'].astype('category')
df['label'].value_counts()

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 8), constrained_layout=True)
ax = ax.flatten()
j = 0
for i in df['label'].unique():
    ax[j].imshow(plt.imread(df[df['label'] == i].iloc[0, 0]))
    ax[j].set_title(i)
    j = j + 1




#  PREPROCESSING
X_train, X_test=train_test_split(df, test_size=0.2, random_state=SEED)

trainGen = ImageDataGenerator(preprocessing_function= preprocess_input, validation_split=0.25 )

testGen =ImageDataGenerator( preprocessing_function= preprocess_input)

X_train_img = trainGen.flow_from_dataframe(dataframe=X_train, x_col='path', y_col='label',
                                           class_mode= class_mode , subset='training',
                                           color_mode= color_mode, batch_size=batch_size, target_size = target_size,
                                           shuffle = shuffle)

X_val_img = trainGen.flow_from_dataframe(dataframe=X_train, x_col='path', y_col='label',
                                         class_mode=class_mode, subset='validation',
                                         color_mode= color_mode, batch_size=batch_size,target_size = target_size,
                                         shuffle = shuffle)

X_test_img =testGen.flow_from_dataframe(dataframe=X_test, x_col='path', y_col='label',
                                        class_mode=class_mode, color_mode=color_mode,
                                        batch_size=batch_size, shuffle=shuffle,target_size = target_size)




#    VISUALIZE DATASET
fit, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
ax = ax.flatten()
j = 0

for _ in range(6):
    img, label = X_train_img.next()
    pixels = np.asarray(img)
    print('Data Type: %s' % img.dtype)
    print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    print(img.shape)
    ax[j].imshow(img[0], )
    ax[j].set_title(label[0])
    j = j + 1




#  IMPORT MODEL AND ADD CUSTOM LAYERS

model = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape= input_shape)
for layer in model.layers[:17]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
prediction = Dense(num_class, activation='softmax')(x)

model_final = Model(inputs = model.input, outputs = prediction)



# COMPILE THE MODEL

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers, metrics=["accuracy"])



# VISUALIZE MODEL
model_final.summary()

# FIT THE MODEL

history = model_final.fit(X_train_img, epochs=epochs, validation_data=X_val_img, verbose=1)




# EVALUATE MODEL
print("Evaluate on test data")
results = model_final.evaluate(X_test_img, batch_size=10)
print("test loss, test acc:", results)

predict_x=model_final.predict(X_test_img)
classes_x=np.argmax(predict_x,axis=1)
pred_df=X_test.copy()
labels={}
for l,v in X_test_img.class_indices.items():
    labels.update({v:l})
pred_df['pred']=classes_x
pred_df['pred']=pred_df['pred'].apply(lambda x: labels[x])







# MATRICE CONFUSION

print(f"Accuracy Score: {accuracy_score(pred_df['label'],pred_df['pred'])}")
sns.heatmap(confusion_matrix(pred_df['label'],pred_df['pred']), annot=True, fmt='2d')

# PLOT ACCURACY AND LOSS




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()





#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()






model_final.save(name_of_model)






