
# coding: utf-8

# ## Create "Model Topper" using example

# In[4]:


import os

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
#from keras import backend as K
from keras import callbacks


# In[5]:

# dimensions of our images.
img_width, img_height = 299, 299

# dimensions of our images.
img_width, img_height = 299, 299
n_features = 1

train_data_dir = 'binary_gender_masked/train/'
validation_data_dir = 'binary_gender_masked/test/'

available_train_files = len(os.listdir(train_data_dir + 'female/'))     + len(os.listdir(train_data_dir + 'male/'))
available_test_files = len(os.listdir(validation_data_dir + 'female/'))     + len(os.listdir(validation_data_dir+'male/'))
    
nb_train_samples  = available_train_files - available_train_files % 8
nb_validation_samples = available_test_files - available_test_files % 8

batch_size = 8


# In[6]:

available_train_files % 8


# Useful metrics for model evaluation

# In[7]:

# def precision(y_true, y_pred):
#     """Precision metric.

#     Only computes a batch-wise average of precision.

#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision


# def recall(y_true, y_pred):
#     """Recall metric.

#     Only computes a batch-wise average of recall.

#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall


# Create Generators

# In[8]:

datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode = 'binary')

test_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode = 'binary')


# Load Standard Xception Model

# In[9]:

model = applications.xception.Xception(include_top=False, weights='imagenet')


# Build My "Custom" Model

# In[10]:

watch_model = Sequential()
watch_model.add(model)
watch_model.add(GlobalAveragePooling2D(name='avg_pool'))
watch_model.add(Dense(1, activation="sigmoid"))
watch_model.summary()


# Freeze Exception Layers

# In[11]:

for layer in watch_model.layers[0].layers:
    layer.trainable = False


# Compile

# In[12]:

watch_model.compile(
    loss = "binary_crossentropy", 
    optimizer='sgd', 
    metrics=["binary_accuracy"])


# In[13]:

#Fit 

epochs = 5

cb1 = [callbacks.ModelCheckpoint(
    'xception_binary_gender_masked_best1.h5',
    monitor='val_loss',
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto', period=1)]




# fine-tune the model
watch_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data = test_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = cb1)

watch_model.save('xception_binary_gender_masked_final1.h5')


# In[14]:

del watch_model


# In[15]:

watch_model = load_model('xception_binary_gender_masked_final1.h5')


# In[16]:

#Unfreeze a few more layers and allow to run

# watch_model.layers[0].summary()

# Freeze convolutional layers
for layer in watch_model.layers[0].layers[-3:]:
    layer.trainable = True

epochs = 20


watch_model.compile(
    loss = "binary_crossentropy", 
    optimizer='sgd', 
    metrics=["binary_accuracy"])

# watch_model.layers[0].summary()

cb2 = [callbacks.ModelCheckpoint(
    'xception_binary_gender_masked_best2.h5',
    monitor='val_loss',
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto', period=1)]

# fine-tune the model
watch_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data = test_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = cb2)


watch_model.save('xception_binary_gender_masked_final2.h5')


# In[17]:

#del watch_model


# In[18]:

#watch_model = load_model('xception_binary_unfrozen_convos1_test.h5')


# In[ ]:



