import matplotlib.pyplot as plt
import tensorflow as tf
import sys, time, os, warnings, re, pickle, os.path, random
import numpy as np
import pandas as pd 
from clr_callback import *
from tqdm import tqdm
from random import randint
from collections import Counter 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import Model
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
print("python {}".format(sys.version))
print("keras version {}".format(tf.keras.__version__));
print("tensorflow version {}".format(tf.__version__))

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95 #Using 95% of the available memory of the GPU
config.gpu_options.visible_device_list = "0"
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class Data():
    
    df = None
    tokenizer = Tokenizer()
    wraped = False
    max_caption_length = 0
    vocab_size = 0
    reverse_word_map = None
    img_embedder = None
    features = dict()
    
    '''Fill up the df field. The column named index keeps 
       track of which version of the five captions that are 
       assigned to each image we're looking at'''
    def fill_text_df(self, path):
        file = open(path,'r')
        text = file.read()
        file.close()
        txtdata = []
        for line in text.split('\n'):
            cols = line.split('\t')
            if len(cols) == 1:
                continue
            w = cols[0].split("#")
            if not w[0] == '2258277193_586949ec62.jpg.1':
                txtdata.append(w + [cols[1]])
        self.df = pd.DataFrame(txtdata,columns=["filename","index","caption"])
    
    '''Keep track of some info about the text data'''
    def get_info(self):
        uni_images = np.unique(self.df.filename.values)
        print("\033[1m Number of unique images: " + str(len(uni_images)) + '\033[0m\n')
        self.vocab_size = len(self.df.caption.str.split(expand=True).stack().value_counts()) + 1
        print("\033[1m Number of unique word tokens: " + str(self.vocab_size) + '\033[0m\n')
        
    '''Show a random img and its five captions'''
    def show_random_img_example(self):
        img_name = self.df.filename.values[randint(0, len(self.df.filename.values))]
        target_size = (256,256,3)
        fig = plt.figure(figsize=(5,5))
        captions = list(self.df["caption"].loc[self.df["filename"]==img_name].values)
        sep = '-' * 100
        print('\033[1m' + img_name + '\033[0m\n' + sep)
        for idx, caption in enumerate(captions):
            print(str(idx + 1) + ": " + caption + '\n' + sep)
        image_load = load_img('Dataset/Flickr_8k_Images/' + '/' + img_name, target_size=target_size)
        plt.axis('off')
        plt.imshow(image_load)
        
        '''Show img'''
    def show_img(self, filename):
        target_size = (256,256,3)
        fig = plt.figure(figsize=(5,5))
        image_load = load_img('Dataset/Flickr_8k_Images/' + '/' + filename, target_size=target_size)
        plt.axis('off')
        plt.imshow(image_load)
        
    '''Clean up the caption column from punctuation and 
       upper case letters, digits etc.'''
    def clean_text(self):
        self.df.caption = (self.df.caption.str.lower()   # lowercase
                           .str.replace(r'[^\w\s]+', '') # rem punctuation
                           .str.replace(r'\b\w\b', '')   # Remove single letter words
                           .str.replace(r'\s+', ' ') 
                           .str.replace('\d+', '')       # Remove numbers
                           .str.strip())                 # rem trailing wh-spaces
        
    '''Plot top 20 most frequent words in a bar plot'''     
    def plot_word_dist(self, count = 20):
        plt.figure(figsize=(20,3))
        self.df.caption.str.split(expand=True).stack().value_counts()[:count].plot(kind='bar')
        plt.title('Top ' + str(count) + ' words with highest frequency')
        plt.show()
        
    '''Add start/end-tokens to each caption'''
    def wrap_captions(self):
        if not self.wraped:
            self.df['caption'] = 'ztartz ' + self.df['caption'] + ' endzz'
            self.wraped = True
      
    def get_max_length(self):
        return max(len(line.split()) for line in list(self.df['caption']))
      
    '''Fit the tokenizer'''
    def fit_tokenizer(self):
        self.tokenizer.fit_on_texts(list(self.df["caption"]))
        self.df['tokens'] = self.tokenizer.texts_to_sequences(self.df['caption'])
        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        
    '''Looking up words in dictionary'''
    def tokens_to_text(self, list_of_tokens):
        words = [self.reverse_word_map.get(letter) for letter in list_of_tokens]
        return(words)
    
    '''Load DenseNet'''
    def load_pretrained_img_embedder(self):
        self.img_embedder = DenseNet201(weights=None)
        self.img_embedder.load_weights("Models/densenet201_weights_tf_dim_ordering_tf_kernels.h5")
        self.img_embedder = Model(inputs=self.img_embedder.inputs, outputs=self.img_embedder.layers[-2].output)
        
    '''Gets called if features are already stored'''
    def load_stored_features(self):
        with open('Dataset/features.pickle', 'rb') as handle:
            self.features = pickle.load(handle)
    
    '''Use DenseNet to extract embeddings from pictures'''
    def generate_and_store_features(self, path):
        if os.path.isfile('Dataset/features.pickle'):
            print ("Features already stored in pickle, loading them instead")
            self.load_stored_features()
            return
        target_size = (224, 224)
        for name in tqdm(os.listdir(path)):
            filename = path + name
            image = load_img(filename, target_size=target_size)
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            feature = self.img_embedder.predict(image, verbose=0)
            self.features[name] = feature.flatten()
        # Store in pickle 
        print ("Storing features as pickle")
        with open('Dataset/features.pickle', 'wb') as handle:
            pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    '''Used when fitting model since all images cant fit in memory'''
    def data_generator(self, data_df, batch_size):
        _count = 0
        images = data_df.filename.unique()
        np.random.shuffle(images)
        while True:
            if _count >= len(images):
                _count = 0
            sequence_batch, img_batch, word_batch = list(), list(), list()
            for i in range(_count, min(len(images), _count + batch_size)):
                image_id = images[i]
                captions = list(data_df.loc[data_df['filename'] == image_id]['tokens'])
                np.random.shuffle(captions)
                for caption in captions:
                    for j in range(1, len(caption)):
                        in_txt = pad_sequences([caption[:j]], maxlen=self.max_caption_length)[0]
                        out_txt = to_categorical(caption[j], num_classes = self.vocab_size)
                        img_batch.append(self.features[image_id].flatten())
                        sequence_batch.append(in_txt)
                        word_batch.append(out_txt)            
            
            _count = _count + batch_size
            sequence_batch = np.array(sequence_batch)
            img_batch = np.array(img_batch)
            word_batch = np.array(word_batch)
            yield ([img_batch, sequence_batch], word_batch)
    
    '''Create caption '''
    def generate_caption(self, model, img):
        in_text = 'ztartz'
        photo = img.flatten()
        for i in range(self.max_caption_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_caption_length)
            yhat = model.predict([np.array(photo[np.newaxis,]), np.array(sequence)], verbose=0)
            yhat = np.argmax(yhat)
            word = self.tokens_to_text([yhat])
            if word is None:
                break
            in_text += ' ' + word[0]
            if word[0] == 'endzz':
                break
        return in_text