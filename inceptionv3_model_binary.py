
# coding: utf-8

# ## Transfer Learning using Xception to Predict Intended Gender of Watch

# In[7]:


import os

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
#from keras import backend as K
from keras import callbacks


# In[8]:

# dimensions of our images.
img_width, img_height = 299, 299

# dimensions of our images.
img_width, img_height = 299, 299
n_features = 1

top_model_weights_path = 'bottleneck_xception_model.h5'
train_data_dir = 'multi_class_testing/train/'
validation_data_dir = 'multi_class_testing/test/'

available_train_files = len(os.listdir(train_data_dir + 'female/'))     + len(os.listdir(train_data_dir + 'male/'))
available_test_files = len(os.listdir(validation_data_dir + 'female/'))     + len(os.listdir(validation_data_dir+'male/'))
    
nb_train_samples  = available_train_files - available_train_files % 8
nb_validation_samples = available_test_files - available_test_files % 8

batch_size = 8


# In[9]:

available_train_files % 8


# Create Generators

# In[10]:

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

# In[11]:

model = applications.InceptionV3(include_top=False, weights='imagenet')


# Build My "Custom" Model

# In[12]:

watch_model = Sequential()
watch_model.add(model)
watch_model.add(GlobalAveragePooling2D(name='avg_pool'))
watch_model.add(Dense(1, activation="sigmoid"))
watch_model.summary()


# In[13]:

watch_model.layers[0].summary()


# Freeze Exception Layers

# In[14]:

for layer in watch_model.layers[0].layers:
    layer.trainable = False


# Compile

# In[15]:

watch_model.compile(
    loss = "binary_crossentropy", 
    optimizer='sgd', 
    metrics=["binary_accuracy"])


# In[20]:

#Fit 

epochs = 10

cb1 = [callbacks.ModelCheckpoint(
    'inceptionv3_binary_frozen_best.h5',
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

watch_model.save('inceptionv3_binary_frozen_final.h5')


# In[17]:

del watch_model


# In[21]:

watch_model = load_model('inceptionv3_binary_frozen_final.h5')


# In[22]:

watch_model.layers[0].summary()


# In[25]:

#Unfreeze a few more layers and allow to run

# watch_model.layers[0].summary()

# Freeze convolutional layers
# for layer in watch_model.layers[0].layers[-3:]:
#     layer.trainable = True

    
for layer in watch_model.layers[0].layers[:249]:
    layer.trainable = False
    
for layer in watch_model.layers[0].layers[249:]:
    layer.trainable = True    
    
    
    
    
epochs = 20


watch_model.compile(
    loss = "binary_crossentropy", 
    optimizer='sgd', 
    metrics=["binary_accuracy"])

# watch_model.layers[0].summary()

cb2 = [callbacks.ModelCheckpoint(
    'inceptionv3_binary_unfrozen_convos1_best.h5',
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


watch_model.save('inceptionv3_binary_unfrozen_convos1_final.h5')


# In[ ]:

#del watch_model


# In[24]:

watch_model = load_model('inceptionv3_binary_unfrozen_convos1_final.h5')


# In[ ]:



