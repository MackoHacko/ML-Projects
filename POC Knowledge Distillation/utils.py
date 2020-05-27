from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Input, Dense, Lambda, concatenate, Dropout, Flatten, Activation
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.special import softmax
import matplotlib.pyplot as plt
import numpy as np
import os

def make_teacher(input_shape, nb_classes):
    teacher = Sequential()
    teacher.add(Flatten(input_shape=input_shape))
    teacher.add(Dense(1024, activation='relu'))
    teacher.add(Dropout(0.2))
    teacher.add(Dense(512, activation='relu'))
    teacher.add(Dropout(0.2))
    teacher.add(Dense(256, activation='relu'))
    teacher.add(Dropout(0.2))
    teacher.add(Dense(128, activation='relu'))
    teacher.add(Dropout(0.2))
    teacher.add(Dense(nb_classes, name = 'logits'))
    teacher.add(Activation('softmax'))

    teacher.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return teacher
    
def small_net(input_shape, nb_classes):
    student = Sequential()
    student.add(Flatten(input_shape=input_shape))
    student.add(Dense(32, activation='relu'))
    student.add(Dense(16, activation='relu'))
    student.add(Dense(nb_classes))
    return student

def knowledge_distillation_loss(y_true, y_pred, nb_classes, alpha = 0.2, beta = 1):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    loss = alpha * logloss(y_true,y_pred) + beta * logloss(y_true_softs, y_pred_softs)
    
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :10]
    y_pred = y_pred[:, :10]
    return categorical_accuracy(y_true, y_pred)