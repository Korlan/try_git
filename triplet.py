#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 07:27:35 2021

@author: korlan
"""



import keras
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image 
from PIL import ImageFilter
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import requests as r
from keras.layers import Input
from keras.layers import concatenate
import matplotlib.pyplot as plt
#plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy #as binary_crossentropy

from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix
#from transformers import TFCamembertModel, CamembertConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import model_from_yaml
from sklearn.metrics import hamming_loss, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score

import random
#from pca_plotter import PCAPlotter
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
def get_images_and_labels(path_to_train):
    #print(os.path.basename(path))
    df = pd.DataFrame()
    columns = ['filename','parentclass','subclass']
    data = []
    images = []
    #labels = []
    path = path_to_train
    IMAGE_DIMS = (256, 256, 3)
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            #print('FILE :', os.path.join(dirpath, f))
            fullname = os.path.join(dirpath, f)
    
            dirname = os.path.dirname(fullname)
            path_dirs = fullname.split("/")
            #print(fullname)
            #print(path_dirs)
            image = cv2.imread(fullname)
            #print(image)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            #image = image.astype("float") / 255.0
            image = img_to_array(image)
            images.append(image)
            try:
                #values = [path_dirs[-1],path_dirs[-3],path_dirs[-2]]
                values = [path_dirs[5],path_dirs[3],path_dirs[4]]
                zipped = zip(columns, values)
                a_dictionary = dict(zipped)
                #print(a_dictionary)
                data.append(a_dictionary)
                #df = pd.concat([path_dirs[4],path_dirs[2],path_dirs[3]],index=['filename','parentclass','subclass']).T
                #print(path_dirs[4], path_dirs[2], path_dirs[3])
            except IndexError:
                #values = [path_dirs[-1], path_dirs[-2]]
                values = [path_dirs[4], path_dirs[3]]
                zipped = zip(columns, values)
                a_dictionary = dict(zipped)
                #print(a_dictionary)
                data.append(a_dictionary)
                #df = pd.concat([path_dirs[3], path_dirs[2]], index = ['filename', 'parentclass']).T
                #print(path_dirs[3], path_dirs[2])
                #print(path_dirs[-1], path_dirs[-2])
    df = df.append(data, True)
    #print(df)
    parentclass = df['parentclass'].tolist()
    subclass = df['subclass'].tolist()
    #print(type(parentclass))
    #print(parentclass)
    #labels = parentclass + subclass
    #print(labels)
    #print(type(labels))
    
    #this part is not necessary, it is for encoding parent and child classes seperately
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(parentclass)
    #print(list(le.classes_))
    parent = df.filter(['filename','parentclass'], axis=1)
    #print(parent)
    child = df.filter(['filename','subclass'], axis=1)
    #print(child)
    level1 = le.transform(parentclass)
    #print(level1)
    
    #multilable binary encoding part
    from sklearn.preprocessing import MultiLabelBinarizer
    #df["subclass"] = df["subclass"].fillna('')
    subset = df[['parentclass', 'subclass']]
    #print(subset)
    #print(type(subset))
    
    tuples = [tuple(x) for x in subset.to_numpy()]
    #print(tuples)
    labels = parentclass + subclass
    labels = np.asarray(labels)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform([labels])
    #print(mlb.classes_)
    classes = mlb.classes_
    #print(classes)
    binary_labels = []
    for item in tuples:
        binary_labels.append(mlb.transform([item]))
    
    binary_df = pd.DataFrame()
    for each in binary_labels:
        #binary_df = pd.DataFrame(each, columns = classes)
        binary_df = binary_df.append(pd.DataFrame(each, columns = classes))
        #mydataframe = mydataframe.append(new_row, ignore_index=True)
    #print(binary_df.head())
    binary_df.insert(0, "filename", df["filename"].values, True)
    binary_df = binary_df.loc[ : , binary_df.columns != 'nan']

    #print(binary_df.head())
    binary_global = binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos', 'Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)
    binary_parent = binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos'], axis=1)
    binary_child = binary_df.filter(['Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)
    
    images = np.array(images, dtype="float") / 255.0
    #print(binary_parent.head())
    y1 = binary_parent.values
    #print(binary_child.loc[[100]])
    y2 = binary_child.values
    global_y = binary_global.values
    #print(y1[100])
    #print(y2[100])
    
    return images, binary_df#y1, y2, global_y
def my_cool_preprocessor(text):
    import re
    import nltk
    from nltk.stem import PorterStemmer
    porter_stemmer=PorterStemmer()
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    #text=re.sub("[^A-Za-z0-9]", "", text)
    text=re.sub("\\s+(le|des|du|de|la|au|et|en|est|dans|les)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)
def get_captions_dataframe():
    import os, json
 
    # this finds our json files
    path_to_json = './database/json/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    
    # here I define my pandas Dataframe with the columns I want to get from the json
    jsons_data = pd.DataFrame(columns=['renderURL', 'caption'])
    all_jsons_data = pd.DataFrame(columns=['renderURL', 'caption'])
    # we need both the json and an index number so use enumerate()
    
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            data = json.load(json_file)
            list_ = []
            for row in data:
                caption = os.path.basename(row['caption'])
                renderURL = os.path.basename(row['renderURL'])
                jsons_data.loc[index] = [renderURL, caption]
                all_jsons_data = all_jsons_data.append(jsons_data, ignore_index = True)
                jsons_data = pd.DataFrame(columns=jsons_data.columns)
    return all_jsons_data
'''l2_normalize = tf.math.l2_normalize
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)'''

'''def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B, A_tensor, B_tensor):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A_tensor, K.permute_dimensions(B_tensor, (0,2,1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den

    return dist_mat'''

'''def triplet_loss(inputs):
        anchor, positive, negative = inputs
        #p_dist = K.sum(K.square(anchor-positive), axis=-1)
        #n_dist = K.sum(K.square(anchor-negative), axis=-1)
        #return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        #p_dist = keras.layers.Lambda(pairwise_cosine_sim)([anchor, positive])
        #cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        #p_dist = cosine_loss(anchor, positive)
        #n_dist = cosine_loss(anchor, negative)
        p_dist = keras.layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, positive])
        n_dist = keras.layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, negative])
        n_dist = keras.layers.Lambda(pairwise_cosine_sim)([anchor, negative])
        return K.sum(p_dist - n_dist)/batch_size'''
'''class TripletLossLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    # convenience l2_norm function

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        #p_dist = keras.layers.Lambda(pairwise_cosine_sim)([anchor, positive])
        #p_dist = keras.layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, positive])
        #n_dist = keras.layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, negative])
        #n_dist = keras.layers.Lambda(pairwise_cosine_sim)([anchor, negative])
        #return K.sum(p_dist - n_dist)/batch_size
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss'''
'''def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

def GetModel(input_shape, embedding_dim):
    base_model = VGG16(weights='imagenet',
                              include_top = False,
                              pooling = 'avg', 
                              #weights = resnet_weights_path
                              )
    # Freeze base model
    #input_layer = Input(shape = (96,96,3))
    #input_image = pre_weights.output
    #base_weights.trainable = False
    #base_model = ResNet50(weights='imagenet', include_top=False, pooling='max')
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Dropout(0.6)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
    embedding_model = Model(base_model.input, x, name="embedding")

    input_shape = input_shape
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
       
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))

    return embedding_model, triplet_model'''
def create_batch(batch_size, random, x_train, y_train):
    x_anchors = np.zeros((batch_size, x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    x_positives = np.zeros((batch_size, x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    x_negatives = np.zeros((batch_size, x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    print(len(y_train[0]))
    x_anchors_labels = np.zeros((batch_size, len(y_train[0])))
    x_positives_labels = np.zeros((batch_size, len(y_train[0])))
    x_negatives_labels = np.zeros((batch_size, len(y_train[0])))
    #x_anchors = np.zeros((batch_size, 196608))
    #x_positives = np.zeros((batch_size, 196608))
    #x_negatives = np.zeros((batch_size, 196608))
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, x_train.shape[0] - 1)
        #print(random_index)
        x_anchor = x_train[random_index]
        x_anchor_label = y_train[random_index]
        #print(x_anchor)
        y = y_train[random_index]
        #print(y)
        
        #print([np.array_equal(y_train[i],y) for i in range(len(y_train))])
        
        indices_for_pos = np.squeeze(np.where([np.array_equal(y_train[i],y) for i in range(len(y_train))]))
        #indices_for_pos = np.squeeze(np.where(y_train==y))
        #print(indices_for_pos.shape)
        
        indices_for_neg = np.squeeze(np.where([np.array_equal(y_train[i],y)==False for i in range(len(y_train))]))
        #indices_for_neg = np.squeeze(np.where(y_train != y))
        #print(indices_for_neg.shape)
        pos_rand = indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]
        x_positive = x_train[pos_rand]
        x_positive_label = y_train[pos_rand]
        #print(rand)
        #print(x_positive)
        #print(x_positive_label)
        #print(x_positive.shape)
        neg_rand = indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]
        x_negative = x_train[neg_rand]
        x_negative_label = y_train[neg_rand]
        #print(x_negative.shape)
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        x_anchors_labels[i] = x_anchor_label
        x_positives_labels[i] = x_positive_label
        x_negatives_labels[i] = x_negative_label
    #print(x_positives.shape)
    #print(x_negatives.shape)    
    return [x_anchors, x_positives, x_negatives], [x_anchors_labels, x_positives_labels, x_negatives_labels]

def plot_triplets(examples):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i], (256, 256, 3)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def build_model(input_shape, margin=0.2):   

     # Define the tensors for the three input images
    #labels_a =Input((4,))
    #labels_p = Input((4,))
    #labels_n = Input((4,))
    
    #second_layer_labels =Input((7,))
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = build_network(anchor_input)
    encoded_p = build_network(positive_input)
    encoded_n = build_network(negative_input)
    #encoded_a = build_network(input_shape)
    #encoded_p = build_network(input_shape)
    #encoded_n = build_network(input_shape)
    level1_a = Dense(4, activation='softmax')(encoded_a)
    level1_p = Dense(4, activation='softmax')(encoded_p)
    level1_n = Dense(4, activation='softmax')(encoded_n)
    #softmax
    #level1 = softmax_output_level1(encoded_a, encoded_p, encoded_n)
    #level2 = softmax_output_level2(encoded_a, encoded_p, encoded_n)
    #TripletLoss Layer
    #loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    #loss_layer = triplet_loss([encoded_a, encoded_p, encoded_n])
    #loss_layer = Lambda(triplet_loss, name='triplet_loss')([encoded_a, encoded_p, encoded_n])
    #loss_layer = triplet_loss()
    #triplet_loss(self, inputs)
    #categorical_crossentropy = CategoricalCrossentropy(from_logits=False)
    #output1_loss_a = categorical_crossentropy(K.flatten(labels_a), K.flatten(level1_a))
    #output1_loss_p = categorical_crossentropy(K.flatten(labels_p), K.flatten(level1_p))
    #output1_loss_n = categorical_crossentropy(K.flatten(labels_n), K.flatten(level1_n))
    #output1_loss_layer = Lambda(output_loss, name='softmax_loss')([output1_loss_a, output1_loss_p, output1_loss_n])
    #output2_loss = categorical_crossentropy(K.flatten(second_layer_labels), K.flatten(level2))
    # Connect the inputs with the outputs
    #inputs = [encoded_a.input, encoded_p.input, encoded_n.input, labels_a, labels_p, labels_n]
    #inputs = [anchor_input, positive_input, negative_input, labels_a, labels_p, labels_n]
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [level1_a, level1_p, level1_n]
    output_tripets = [encoded_a, encoded_p, encoded_n]
    output_triplet_loss = K.mean(triplet_loss(output_tripets))
    #output_softmax = [output1_loss_a, output1_loss_p, output1_loss_n]
    #output_loss_softmax = K.mean(output_loss(output_softmax))
    #triplet_model = Model(inputs, outputs)
    #triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    #network_train = Model(inputs=[encoded_a.input, encoded_p.input, encoded_n.input, labels_a, labels_p, labels_n],outputs=outputs)
    network_train = Model(inputs=inputs,outputs=outputs)
    network_train.add_loss(output_triplet_loss)
    network_train.add_metric(output_triplet_loss, name='output_triplet_loss', aggregation='mean')
    #network_train.add_loss(output_loss_softmax)
    #network_train.add_metric(output_loss_softmax, name='output_loss_softmax', aggregation='mean')
    
    #network_train.add_loss([output1_loss_a, output1_loss_p, output1_loss_n])
    #network_train.add_loss(output2_loss)
    #network_train.add_loss(loss_layer)
    network_train.summary()
    # return the model
    return network_train
def output_loss(inputs):
    anchor, positive, negative = inputs
    loss = K.log(anchor) + K.log(positive) + K.log(negative)
    #loss = (-1) * loss / batch_size
    return K.mean(loss)

def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    distances = []
    for i in range(len(anchor) - 1):
        dist = K.square(anchor[i] - anchor[i+1])
        distances = dist
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    treshold = K.mean(negative_distance)
    average = K.mean(distances)
    alpha = 0.2 + treshold - average
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, alpha + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)
def build_network(input_):
    #from keras.models import Model
    #from keras.layers import Input
    #from keras.layers import Dense
    #from keras.layers import Flatten
    #from keras.layers.convolutional import Conv2D
    
    #visible = Input(shape=input_shape)
    conv1 = Conv2D(20, kernel_size=3, activation='relu')(input_)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #conv2 = Conv2D(30, kernel_size=3, activation='relu')(conv1)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #conv3 = Conv2D(30, kernel_size=3, activation='relu')(conv2)
    conv4 = Conv2D(15, kernel_size=3, activation='relu')(conv1)
    flat = Flatten()(conv4)
    hidden1 = Dense(20, activation='relu')(flat)
    #network = Model(inputs = visible, outputs = hidden1)
    #network.summary()
    #hidden1 = Dense(10, activation='relu')(flat)
    #output = Dense(1, activation='sigmoid')(hidden1)
    
    return hidden1

def run_test():
    #######################################################################################
    #get images and its labels: TRAIN
    path_to_train = "./database/database for hierarchical classification 6 classes_train/"
    images, binary_df = get_images_and_labels(path_to_train)

    #######################################################################################
    #get images and its labels: TEST
    path_to_test = "./database/database for hierarchical classification 6 classes_test/"
    test_images, test_binary_df = get_images_and_labels(path_to_test)
    
    #######################################################################################
    #cross-validation
    num_folds = NUM_FOLD
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
    #kfold = KFold(n_splits=num_folds, shuffle=True)
    
    #######################################################################################
    #merge inputs and targets
    inputs = np.concatenate((images, test_images), axis=0)
    targets = pd.concat([binary_df, test_binary_df])
    targets.insert(0, 'index_col', range(0, len(targets)), True)
    #targets.reset_index(inplace=True, drop=True)
    
    #######################################################################################
    #get captions
    all_jsons_data = get_captions_dataframe()
    
    #######################################################################################
    #drop duplicated samples
    all_jsons_data.drop_duplicates(subset ="renderURL", keep = False, inplace = True)
    
    #targets.insert(-1, 'caption', np.where(targets["filename"].isin(all_jsons_data['renderURL']) == True, all_jsons_data['caption'], 'nan'), True)
    #print(np.where(targets["filename"].isin(all_jsons_data['renderURL']) == False))
    #merged.drop_duplicates(subset ="filename", keep = False, inplace = True)
    
    #######################################################################################
    #merge two dataframe without dropping the rows that are not the same = (for 1200)
    merged = targets.merge(all_jsons_data, how='left', left_on='filename', right_on='renderURL')
    merged['caption'] = merged['caption'].replace(np.nan, "-1")
    #    test_binary_df['caption'] = test_binary_df['caption'].replace(np.nan, "")
    
    #######################################################################################
    #get indexes of no caption rows from dataframe, and select tensors of images by this indexes  
    indexes_no_caption_from_dataframe = np.asarray(np.where(targets["filename"].isin(all_jsons_data['renderURL']) == False))
    indexes = indexes_no_caption_from_dataframe.flatten()
    #indexes = indexes.reshape(indexes.shape[0],-1)
    
    inputs_with_caption = inputs
    #inputs_with_caption = tf.gather_nd(inputs_with_caption, indexes)
    inputs_with_caption = np.delete(inputs_with_caption, indexes, axis=0)

    #######################################################################################   
    #merge two dataframe dropping the rows that are not the same = (for 500)
    drop_merged = targets.merge(all_jsons_data, left_on='filename', right_on='renderURL')
    
    #drop_merged.drop_duplicates(subset ="filename", keep = False, inplace = True)
    #drop_merged.sort_index()
    #drop_merged.reset_index(inplace=True, drop=True)
    
    #######################################################################################   
    #reset index
    new_index = pd.Series(range(0, len(drop_merged)), name='index_col')
    drop_merged.update(new_index)

    #######################################################################################   
    #save dataframes:
    #merged - with caption, exist empty captions (1200 samples)
    #drop_merged - with caption, no empty captions (500 samples)
    #targets - no caption
    merged.to_csv('csv/merged.csv', index=False)
    drop_merged.to_csv('csv/drop_merged.csv', index=False)
    targets.to_csv('csv/targets.csv', index=False)

    #######################################################################################
    #get the caption values
    #all_caption = drop_merged.caption.values
    all_caption = merged.caption.values
    list_caption = drop_merged.caption.values
    #######################################################################################
    
    #######################################################################################
    #BOW through count vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    #count_vect = CountVectorizer(ngram_range=(1, 2), preprocessor=my_cool_preprocessor)
    count_vect = CountVectorizer(preprocessor=my_cool_preprocessor)
    #count_vect = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.50, preprocessor=my_cool_preprocessor)
    count_vect.fit(all_caption)
    
    X_train_counts = count_vect.transform(all_caption)    
    print(X_train_counts.shape)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    #tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    #X_train_tf = tf_transformer.transform(X_train_counts)
    #print(X_train_tf.shape)
    
    tfidf_transformer = TfidfTransformer()
    all_caption = tfidf_transformer.fit_transform(X_train_counts).todense()
    print(all_caption.shape)
        
    #######################################################################################   
    #fold number
    fold_no = 1
    
    
    from sklearn.preprocessing import LabelEncoder
    #binary_global = merged.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos', 'Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)
    binary_global = merged.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos'], axis=1)
    global_labels = binary_global.values
    print(global_labels)
    def get_new_labels(y):
        y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
        return y_new
    
    y_new = get_new_labels(global_labels)
    print(y_new)
        
    for train, test in kfold.split(inputs, y_new):
        #######################################################################################
        #get the dataframes
        #print(train)
        print('------------------------------------------------------------------------')
        #print('------------------------------------------------------------------------')
        #print(f"Fold {fold_no}!")
        print('train -  {}   |   test -  {}'.format(np.bincount(y_new[train]), np.bincount(y_new[test])))
        binary_df = merged.iloc[train]
        test_binary_df = merged.iloc[test]
        #print(binary_df['index_col'])
        #print(LabelEncoder().inverse_transform[train])
        #######################################################################################
        #replace the empty caption
        #binary_df['caption'] = binary_df['caption'].replace(np.nan, "")
        #test_binary_df['caption'] = test_binary_df['caption'].replace(np.nan, "")
        
        #binary_df.to_csv('replace.csv', index=False)
        #train_image_indexes = train.reshape(train.shape[0],-1)
        
        #######################################################################################
        #get the images
        train_image = inputs[train]
        print(train_image.shape)
        test_image = inputs[test]
        print(test_image.shape)
        
        #train_image = tf.gather_nd(inputs_with_caption, train_image_indexes)
        #test_image = inputs_with_caption[test]

        #######################################################################################
        #seperate the labels by levels
        binary_global = binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos', 'Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)
        binary_parent = binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos'], axis=1)
        #binary_child = binary_df.filter(['Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)

        #test_binary_global = test_binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos', 'Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)
        #test_binary_parent = test_binary_df.filter(['Cross section', 'Maps', 'Graphs and Tables', 'Photos'], axis=1)
        #test_binary_child = test_binary_df.filter(['Well section', 'Geology', 'One time', 'Permanent', 'Results', 'Description', 'Photos'], axis=1)

        #######################################################################################
        #get the caption values
        train_caption = all_caption[train]#binary_df.caption.values
        print(train_caption.shape)
        test_caption = all_caption[test]#test_binary_df.caption.values
        print(test_caption.shape)
        
        caption_exist = test_binary_df.filter(['caption']).values
        #print(caption_exist)

        #######################################################################################
        #get the labels values
        #level1_labels = binary_parent.values
        #level2_labels = binary_child.values
        #global_labels = binary_global.values
        final_labels = binary_parent.values
        
        #test_level1_labels = test_binary_parent.values
        #test_level2_labels = test_binary_child.values
        #test_global_labels = test_binary_global.values
        #test_final_labels = test_binary_global.values

        #######################################################################################
        #split to train/validation
        #print('------------------------------------------------------------------------')
        #print(f"Training for fold {fold_no} ...")
        
        split = train_test_split(train_image, train_caption, final_labels, test_size=0.2, random_state=42)
        (trainX, validX, trainCaption, validCaption, trainFinal, validFinal) = split

        #######################################################################################
        #define the model
        #print(train_image.shape)
        #print(train_image.shape[0])import tensorflow as tf

        #trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]*trainX.shape[2]*trainX.shape[3]))
        #validX = np.reshape(validX, (validX.shape[0], validX.shape[1]*validX.shape[2]*validX.shape[3]))
       
        #print(trainX.shape)
        #print(final_labels)
        #plot_triplets([train_image[0], train_image[1], train_image[2]])
        #train_dataset, train_labels = create_batch(batch_size, random, trainX, trainFinal)
        #valid_dataset, valid_labels = create_batch(batch_size, random, validX, validFinal)
        #train_dataset = create_batch(batch_size, random, trainX, trainFinal)
        #valid_dataset = create_batch(batch_size, random, validX, validFinal)
        #train_anchor, train_positive, train_negative, train_anchor_labels, train_positive_labels, train_negative_labels = create_batch(batch_size, random, trainX, trainFinal)
        #valid_anchor, valid_positive, valid_negative, valid_anchor_labels, valid_positive_labels, valid_negative_labels = create_batch(batch_size, random, validX, validFinal)
        #print(examples)
        #plot_triplets(examples)
        #embedding_dim = 64
        #emb_size = embedding_dim
        #alpha = 0.2
        #embedding_model, triplet_model = GetModel(input_shape, embedding_dim)
        triplet_model = build_model(input_shape)
        steps_per_epoch = int(trainX.shape[0]/batch_size)

        triplet_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01),metrics=["accuracy"])
        dummy = np.zeros((batch_size, len(3*trainX[0]))),
        def data_generator(batch_size, trainX, trainFinal):
            while True:
                x, y = create_batch(batch_size, random, trainX, trainFinal)
                #y = np.zeros((batch_size, len(3*trainX[0])))
                yield x, y
        history = triplet_model.fit(#x = train_dataset,
                                    #y = train_labels,
                                    data_generator(batch_size, trainX, trainFinal),
                                    #x = [train_anchor, train_positive, train_negative, train_anchor_labels, train_positive_labels, train_negative_labels],
                                    
                              #y = [trainLevel1, trainLevel2],
                              validation_data=(
                                  
                                  #valid_dataset, valid_labels
                                  data_generator(batch_size, validX, validFinal)
                                  #[valid_anchor, valid_positive, valid_negative, 
                                  # valid_anchor_labels, valid_positive_labels, valid_negative_labels],
                                  #dummy
                                  #[validLevel1, validLevel2]
                                  ),
                              epochs=EPOCHS, 
                              verbose=verbose, 
                              batch_size=batch_size,
                              #workers=4,
                              steps_per_epoch=steps_per_epoch, 
                              validation_steps=20,
                              use_multiprocessing=True
                              )
        
        
        fold_no = fold_no + 1

    #######################################################################################
    #confusion matrices  
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)
batch_size = 16
EPOCHS = 1
verbose = 1
NUM_FOLD = 10
run_test()