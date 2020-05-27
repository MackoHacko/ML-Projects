import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dropout
from keras.models import Model, Sequential
from skimage import color
from keras.regularizers import l2
import tensorflow as tf
import pickle
import keras
import sklearn.neighbors as nn
import cv2 as cv

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
    x = x.reshape((x.shape[0], 64, 64, 3))
    return x

def load_labels(input_file):
    d = unpickle(input_file)
    x = d['labels']
    return x

def soft_encoding(img, nn_finder):
    img = color.rgb2lab(img)
    img = cv.resize(img, (16, 16), cv.INTER_CUBIC)
    h, w = img.shape[:2]
    a = np.ravel(img[:, :, 1])
    b = np.ravel(img[:, :, 2])
    ab = np.vstack((a, b)).T
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    y = np.zeros((ab.shape[0], 313))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, 313)
    return y

def gen_input(img):
    img = color.rgb2lab(img)
    inp = img[:,:,0]-50 # Mean centering
    return np.reshape(inp, (64,64,1))

def net_builder(conf):
    '''
    Conf specifies model architechture where 1 row of conf: = 
    [# filters, stride, dialation, BatchNorm_bool, upsampling_bool]

    '''
    k_reg = 1e-3
    model = Sequential()
    for layer in range(len(conf)):          
        if (conf[layer][4]):
            model.add(UpSampling2D(size=(2, 2)))    
        if (layer == 0):         
            model.add(Conv2D(conf[layer][0], (3,3), 
             activation='relu',
             strides = conf[layer][1],
             input_shape = (64,64,1),
             dilation_rate=conf[layer][2], 
             padding='same', 
             kernel_regularizer=l2(k_reg),
             kernel_initializer='glorot_normal'))
        else:
            model.add(Conv2D(conf[layer][0], (3,3), 
             activation='relu',
             strides = conf[layer][1],
             dilation_rate=conf[layer][2], 
             padding='same', 
             kernel_regularizer=l2(k_reg),
             kernel_initializer='glorot_normal'))                        
        if (conf[layer][3]):
            model.add(BatchNormalization())
        model.add(Dropout(0.4))
    model.add(Conv2D(313, (1, 1), activation='softmax', padding='same'))     
    return model

def annealed_mean(dist):
    annealed = np.exp(np.log(dist/0.38 + 1e-8)) # 1e-8 in order to avoid inf
    annealed /= np.sum(annealed)
    return annealed

def decode_prediction(pred):
    q_ab = np.load("Resources/pts_in_hull.npy")
    decoding = np.zeros((16,16,2))
    for i in range(16):
        for j in range(16):
            decoding[i,j,:] = np.sum([np.multiply(x,q_ab[i]) for i, x in enumerate(annealed_mean(pred[0,i,j]))], axis = 0)
    return decoding

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, usage, length, batch_size=600, dim=(64,64,1), ydim=(16,16,313), shuffle=True):
        'Initialization'
        self.dim = dim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = length
        self.on_epoch_end()
        w = 1/(0.5*np.load("Resources/prior_probs.npy").astype(np.float32) + (0.5/313))
        self.weights = tf.convert_to_tensor(w/154.83514)
        self.q_ab = np.load("Resources/pts_in_hull.npy")
        self.nn_finder = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.q_ab)
        if (usage == 'train'):
            self.data = load_data('Data/train_data_batch_1')[:length]
        else:
            self.data = load_data('Data/val_data')[:length]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def multinomal_cross_entropy(self, z, z_hat): 
        cross_entropy = tf.reduce_sum(tf.multiply(z, tf.log(z_hat + 10e-30)), 3)
        idx = tf.argmax(z, axis=3)
        weight = tf.gather(self.weights, idx)
        loss = -(tf.math.multiply(weight, cross_entropy))
        loss = tf.math.reduce_sum(loss, axis = None) / (self.batch_size * (16 * 16))
        return loss
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.ydim))

        # Generate data
        for i, img in enumerate(self.data[indexes]):
            # Store sample
            X[i] = gen_input(img)
            # Store class
            y[i] = soft_encoding(img, self.nn_finder)

        return X, y