import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping,Callback


class hdfRead(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.all_objs = []
        self.all_groups =[]
        self.all_datasets =[]
        
    def list_contents(self, f):
        all_objs = []
        f.visit(all_objs.append)
        all_groups = [ obj for obj in all_objs if isinstance(f[obj],h5py.Group) ]
        all_datasets = [ obj for obj in all_objs if isinstance(f[obj],h5py.Dataset) ]
        return all_groups, all_datasets


class PrintDot(Callback):
      def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print(' ')
            print('.', end='')

            
def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def model(X_train):
    model = Sequential()

    model.add(Conv2D(16, (5,5), padding='same', activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
    model.add(Conv2D(16, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
    model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    return model


def model_1(X_train_1):
    image_input = Input(shape=(X_train_1.shape[1],X_train_1.shape[2],X_train_1.shape[3]))
    other_data_input = Input(shape=(1,))

    conv1 = Conv2D(16, (5,5), padding = 'same', activation='relu')(image_input)
    conv2 = Conv2D(16, (5,5), padding = 'same', activation='relu')(conv1)
    conv2 = MaxPooling2D(pool_size=(4,4))(conv2)
    conv2 = Dropout(0.3)(conv2)

    conv3 = Conv2D(32, (5,5), padding = 'same', activation='relu')(conv2)
    conv4 = Conv2D(32, (5,5), padding = 'same', activation='relu')(conv3)
    conv4 = MaxPooling2D(pool_size=(4,4))(conv4)
    conv4 = Dropout(0.3)(conv4)

    conv5 = Conv2D(64, (5,5), padding = 'same', activation='relu')(conv4)
    conv6 = Conv2D(64, (5,5), padding = 'same', activation='relu')(conv5)
    conv6 = MaxPooling2D(pool_size=(4,4))(conv6)
    conv6 = Dropout(0.3)(conv6)

    first_part_output = Flatten()(conv6)
    dense1 = Dense(32, activation= 'relu')(first_part_output)
    merged_model = concatenate([dense1, other_data_input])
    dense2 = Dense(32, activation= 'relu')(merged_model)
    dense3 = Dense(32, activation= 'relu')(dense2)
    dense4 = BatchNormalization()(dense3)
    dense4 = Dropout(0.3)(dense4)

    predictions = Dense(1)(dense4)

    model_1 = Model(inputs=[image_input, other_data_input], outputs=predictions)
    
    return model_1


def grad_cam_model_2(input_model, x, layer_name, weight, blur):
    
    preprocessed_input = np.expand_dims(x, axis=0)
    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, x.shape[:2], cv2.INTER_LINEAR)
    cam  = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)  ## colorbar()
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    
    ori_img = Image.fromarray((norm(x[:,:,0])*255).astype('uint8')).convert('RGB')
    combined_img = cv2.addWeighted(np.asarray(ori_img), weight, rgb_cam, 1-weight, blur)
    
    return ori_img, heatmap, combined_img


def grad_cam_model_2_1(input_model, input_1, input_2, layer_name, weight, blur):
    
    preprocessed_input_1 = np.expand_dims(input_1, axis=0)
    preprocessed_input_2 = np.expand_dims(np.expand_dims(input_2, axis=0),axis=0)
    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([preprocessed_input_1,preprocessed_input_2])
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, input_1.shape[:2], cv2.INTER_LINEAR)
    cam  = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)  ## colorbar()
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    ori_img = Image.fromarray((norm(input_1[:,:,0])*255).astype('uint8')).convert('RGB')
    combined_img = cv2.addWeighted(np.asarray(ori_img), weight, rgb_cam, 1-weight, blur)

    return ori_img, heatmap, combined_img