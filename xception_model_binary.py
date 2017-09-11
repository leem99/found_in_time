
# coding: utf-8

# ## Create "Model Topper" using example

# In[23]:


import os

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
from keras import backend as K
from keras import callbacks


# In[14]:

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


# In[15]:

available_train_files % 8


# Useful metrics for model evaluation

# In[16]:

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Create Generators

# In[17]:

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

# In[18]:

model = applications.xception.Xception(include_top=False, weights='imagenet')


# Build My "Custom" Model

# In[19]:

watch_model = Sequential()
watch_model.add(model)
watch_model.add(GlobalAveragePooling2D(name='avg_pool'))
watch_model.add(Dense(1, activation="sigmoid"))
watch_model.summary()


# Freeze Exception Layers

# In[20]:

for layer in watch_model.layers[0].layers:
    layer.trainable = False


# Compile

# In[21]:

watch_model.compile(
    loss = "binary_crossentropy", 
    optimizer='sgd', 
    metrics=["accuracy"])


# In[ ]:

#Fit 

epochs = 10

cb1 = [callbacks.ModelCheckpoint(
    'xception_binary_convos1_best.h5',
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

watch_model.save('xception_binary.h5')

#Unfreeze a few more layers and allow to run

watch_model.layers[0].summary()

# Freeze convolutional layers
for layer in watch_model.layers[0].layers[-3:]:
    layer.trainable = True

epochs = 50


cb2 = [callbacks.ModelCheckpoint(
    'xception_binary_unfrozen_convos1_best.h5',
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


watch_model.save('xception_binary_unfrozen_convos1_final.h5')


# In[ ]:



