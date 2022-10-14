import os
from os import path, listdir
from os.path import join, isfile
from glob import glob
from pathlib import Path
import cv2
import tensorflow as tf

import random
import numpy as np
import pandas as pd
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from keras import optimizers

# Define the pathes for the directories of defect classes

Faded_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Faded") 
Covered_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Covered")
Scribbled_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Scribbled")
No_error_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "No_error")
Covered_Faded_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Covered_Faded") 
Covered_Scribbled_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Covered_Scribbled")
Faded_Scribbled_path = os.path.join("/home/keplab/Storage/Personal/Amr/Datasets/Merged_dataset_8", "Faded_Scribbled")


#The listdir function takes a pathname and returns a list of the contents of the directory.
#listdir returns both files and folders, with no indication of which is which. 
#You can use list filtering and the isfile function of the os.path module to separate the files from the folders.
#isfile takes a pathname and returns 1 if the path represents a file, and 0 otherwise.

# To read all images in each defect class with paths
Faded = [os.path.join(Faded_path, f) for f in listdir(Faded_path) if isfile(join(Faded_path, f))]
Covered = [os.path.join(Covered_path, f) for f in listdir(Covered_path) if isfile(join(Covered_path, f))]
Scribbled = [os.path.join(Scribbled_path, f) for f in listdir(Scribbled_path) if isfile(join(Scribbled_path, f))]
No_error = [os.path.join(No_error_path, f) for f in listdir(No_error_path) if isfile(join(No_error_path, f))]
Covered_Faded = [os.path.join(Covered_Faded_path, f) for f in listdir(Covered_Faded_path) if isfile(join(Covered_Faded_path, f))]
Covered_Scribbled = [os.path.join(Covered_Scribbled_path, f) for f in listdir(Covered_Scribbled_path) if isfile(join(Covered_Scribbled_path, f))]
Faded_Scribbled = [os.path.join(Faded_Scribbled_path, f) for f in listdir(Faded_Scribbled_path) if isfile(join(Faded_Scribbled_path, f))]

# To print how many samples in each defect class
print(f"Total Faded: {len(Faded)}")
print(f"Total Covered: {len(Covered)}")
print(f"Total Covered: {len(Scribbled)}")
print(f"Total Covered: {len(No_error_path)}")
print(f"Total Faded: {len(Covered_Faded)}")
print(f"Total Covered: {len(Covered_Scribbled)}")
print(f"Total Covered: {len(Faded_Scribbled)}")

# Deine a dataset array to store all class labels and paths of images for all classes
dataset = []    
for file in Faded:    
    dataset.append([Path(Faded[0]).parent.name, file])
   
for file in Covered:    
    dataset.append([Path(Covered[0]).parent.name, file])

for file in Scribbled:    
    dataset.append([Path(Scribbled[0]).parent.name, file])
for file in No_error:    
    dataset.append([Path(No_error[0]).parent.name, file])

for file in Covered_Faded:    
    dataset.append([Path(Covered_Faded[0]).parent.name, file])
   
for file in Covered_Scribbled:    
    dataset.append([Path(Covered_Scribbled[0]).parent.name, file])

for file in Faded_Scribbled:    
    dataset.append([Path(Faded_Scribbled[0]).parent.name, file])
        

# set_option expand output display to see more columns   
pd.set_option('max_colwidth', 500)

#DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. 

# Define a Dataframe to create your dataset
df = pd.DataFrame(dataset)
# Define the columns of the Dataframe (first column for label and second column for image path)
df.columns = ['Class', 'Path']
# To know how many classes 
total_labels = len(set(df['Class'].values))
# To return the values (names) of the defect classes
labels = set(df['Class'].values)
print(labels)

print(f"Total number of labels: {total_labels}")

# To print inforation about your dataset
print(df.info())

#df.sample(n=5)
#Viewing some samples


# To define the input shape for an image (width, hieght, #of channels)
image_shape = (128, 128, 3)

def resize_image(img_array):
    img = Image.fromarray(img_array)
    img = img.resize(image_shape[:-1])
    #img = img.resize(image_shape)
    return np.array(img)

def show_images(images, title=""):
    fig, ax = plt.subplots(1, len(images), figsize=(10, 10), dpi=100)   
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].set_title(title)
    [x.axis('off') for x in ax]
    plt.show()
    
def load_images_show(image_paths):
    images = [np.array(Image.open(img).convert('RGB')) for img in image_paths]
    images = [resize_image(img) for img in images]    
    return np.array(images)
   

# View random 3 samples from each Sign label (class)
for l in labels:
    Sign_imgs = df[df['Class']==l]['Path']
    Sign_imgs = load_images_show(Sign_imgs.values)        
    show_images(random.sample(list(Sign_imgs), 3), l)

#Train test split from dataset
#We will select 10 images from each class and the rest of the images will be use as test set.
train_list = []
test_list = []

for l in labels:   
    Sign_imgs = df[df['Class']==l]    
    print(l)
    df_train, df_test = train_test_split(Sign_imgs, test_size=0.2,random_state=0)  
    train_list.append(df_train)
    test_list.append(df_test)
# labels_list = []
# for l in labels:
#     labels_list.append(l)
# print(labels_list)

# sss = StratifiedShuffleSplit( n_splits=1, test_size=0.5, random_state=0)
# sss.get_n_splits(Sign_imgs,labels_list)
# print(sss)
# for train_index, test_index in sss.split(Sign_imgs,labels_list):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     train_list.append(X_train)
#     test_list.append(X_test)

# pd.concat() can be used for a simple concatenation of Series or DataFrame objects, 
# just as np.concatenate() can be used for simple concatenations of arrays: 

df_X_train = pd.concat(train_list)   
df_X_test = pd.concat(test_list)

#Verifying split results

# Train samples
df_X_train.groupby('Class').count()

# Test samples
df_X_test.groupby('Class').count()


class Data_Generator(object):
    
    def __init__(self, df, image_shape, batch_size):
        # Prepare parameters
        self.df = df.copy()
        self.h, self.w, self.c = image_shape
        self.batch_size = batch_size
        self.labels = list(set(self.df['Class']))
    
    # def resize_image(self, img_array):
    #     img = Image.fromarray(img_array)
    #     img = img.resize(image_shape[:-1])
    #     #img = img.resize(image_shape)
    #     return np.array(img)
    
    # def load_image(self, url):
    #     img = Image.open(url).convert('RGB')
    #     img = np.array(img)
    #     img = self.resize_image(img)        
    #     return img

    def load_image(self, url):
        img = cv2.imread(url, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128, 128))       
        img = np.array(img, dtype="float")/255
            
        return img
    
    
    def histogram_equalization(self, image):
        b,g,r = cv2.split(image)
        #r = cv2.histogram_equalization(r)
        #g = cv2.histogram_equalization(g)
        #b = cv2.histogram_equalization(b)
        img = np.stack((b,g,r), -1)
        img = np.divide(img, 255)
        return img
    
    def get_batch(self):
        
        while True:            
            # Create holder for batches
            pairs = [np.zeros((self.batch_size, self.h, self.w, self.c)) for i in range(2)]
            targets = np.zeros((self.batch_size,))
            targets[self.batch_size // 2:] = 1  # half are positive half are negative
            random.shuffle(targets)
            
            
            for b in range(self.batch_size):
                # Select anchor image
                selected_label = np.random.choice(self.labels, 1)[0]
                #print(selected_label)

                # Negative - 0 (Different images), Positive = 1 (Same images)
                if targets[b] == 0:
                    # Negative examples
                    labels_ = self.labels.copy()
                    labels_.remove(selected_label) 
                    target_label = np.random.choice(labels_, 1, replace=False)[0]                                   
                                        
                    # load images into pairs
                    image1 = self.df[self.df["Class"] == selected_label].sample(n=1)['Path'].values[0]
                    #print(image1)
                    image1 = self.load_image(image1)
                    image1 = self.histogram_equalization(image1)
                    
                    image2 = self.df[self.df["Class"] == target_label].sample(n=1)['Path'].values[0]
                    #print(image2)
                    image2 = self.load_image(image2)
                    image2 = self.histogram_equalization(image2)
                    
                    pairs[0][b, :, :, :] = image1
                    pairs[1][b, :, :, :] = image2
                else:
                    # Positive examples
                    images = self.df[self.df['Class'] == selected_label].sample(n=2)['Path'].values
                    image1 = self.load_image(images[0])
                    image1 = self.histogram_equalization(image1)
                    
                    image2 = self.load_image(images[1])
                    image2 = self.histogram_equalization(image2)
                    
                    pairs[0][b, :, :, :] = image1
                    pairs[1][b, :, :, :] = image2
            yield pairs, targets.astype(int)
            
# Visualise our generator
# This is ensure our generator is working as espected

batch_size = 6

train_gen = Data_Generator(df, image_shape, batch_size)
batch, targets = next(train_gen.get_batch())


mLabels = ["Different", "Same"]
for n in range(batch_size):
    show_images(random.sample([batch[0][n], batch[1][n]], 2), mLabels[targets[n]])
    


#Create our model SiameseNet

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications import ResNet50V2, VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.backend.tensorflow_backend import set_session
from keras.engine.saving import model_from_json
from keras.layers import Input, Dropout, Conv2D,GlobalMaxPool2D, MaxPooling2D, Concatenate, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Model, load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras import models
from keras import layers
import keras

from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')


#vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=[64, 64, 3])

# # Create the model
# classifer_model = models.Sequential()

# # Add the vgg convolutional base model
# classifer_model.add(vgg_conv)

# # Popping last 3 layers that are added for classification
# # classifer_model.layers.pop()
# # classifer_model.layers.pop()
# # classifer_model.layers.pop()
# # classifer_model.layers.pop()
# classifer_model.summary()

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

    
def create_Model():
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(5120, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)  
        
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(1e-4))
    print(siamese_net.summary())
    
    # return the model
    return siamese_net
def create_Model_pooling():
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    #model.add(Flatten())
    #model.add(Dense(5120, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)  

    encoded_l_Max = MaxPooling2D()(encoded_l)
    encoded_l_Max = Flatten()(encoded_l_Max)    
    encoded_r_Max = MaxPooling2D()(encoded_r)
    encoded_r_Max = Flatten()(encoded_r_Max)  

    encoded_l_Ave = AveragePooling2D()(encoded_l)
    encoded_l_Ave = Flatten()(encoded_l_Ave) 
    encoded_r_Ave = AveragePooling2D()(encoded_r)
    encoded_r_Ave = Flatten()(encoded_r_Ave) 
        
    # # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance_Max = L1_layer([encoded_l_Max, encoded_r_Max])
    L1_distance_Ave = L1_layer([encoded_l_Ave, encoded_l_Ave])

    L1_distance = Concatenate(name ='Concatenate')([encoded_l_Max,encoded_l_Ave,L1_distance_Max, L1_distance_Ave])
    
    FC_1 = Dense(10240,  activation='relu')(L1_distance)
    FC_1 = Dropout(0.2)(FC_1)

    FC_1 = Dense(2560,  activation='relu')(FC_1)
    FC_1 =Dropout(0.2)(FC_1)

    FC_1 = Dense(1024,  activation='relu')(FC_1 )
    FC_1 = Dropout(0.2)(FC_1)
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    
    prediction = Dense(1, activation='sigmoid')(FC_1)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(1e-4))
    print(siamese_net.summary())
    
    # return the model
    return siamese_net

def create_FC_model():
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(5120, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    print("******************")
    print(encoded_l)
    print("******************")
    # max_encoded_l  = encoded_l.reshape(-1,encoded_l[1])
    # max_encoded_r  = encoded_r[-1:]
    # print(max_encoded_l.shape)
   
    print(encoded_l)

   
        
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])


   
    concatenation_layer = Concatenate(name ='Concatenate')([encoded_l, encoded_r])
    FC_1 = Dense(10240,  activation='relu')(concatenation_layer)
    FC_1 = Dropout(0.2)(FC_1)

    FC_1 = Dense(2560,  activation='relu')(FC_1)
    FC_1 =Dropout(0.2)(FC_1)

    FC_1 = Dense(1024,  activation='relu')(FC_1 )
    FC_1 = Dropout(0.2)(FC_1)

    L1_distance = Concatenate(name ='Concatenate2')([L1_distance, FC_1])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(1e-4))
    print(siamese_net.summary())
    
    # return the model
    return siamese_net

def create_FC_VGG_Amr_model(lr=1e-4):
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    # model = Sequential()        
    # pretrain_model = Model(inputs=classifer_model.input, outputs=classifer_model.get_output_at(-1))
    # for layer in pretrain_model.layers: # Set all layers to be trainable
    #     layer.trainable = False
    # for layer in pretrain_model.layers[-4:]: # last 4 layer freeze
    #     layer.trainable = True     
    
    # model.add(pretrain_model)
    # model.add(Flatten())
    # model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    print("[Info] loading imagenet weights...")
    baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(128, 128, 3)))

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(4096, activation='relu')(headModel)
    #headModel = Dropout(0.2)(headModel)
    #headModel = Dense(4096, activation='sigmoid')(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = True
    
    model.summary()
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    concatenation_layer = Concatenate(name ='Concatenate')([encoded_l, encoded_r])
    FC_1 = Dense(4096,  activation='relu')(concatenation_layer)
    FC_1 = Dropout(0.2)(FC_1)

    FC_1 = Dense(1024,  activation='relu')(FC_1)
    FC_1 =Dropout(0.2)(FC_1)

    FC_1 = Dense(256,  activation='relu')(FC_1 )
    #FC_1 = Dropout(0.2)(FC_1)

    L1_distance = Concatenate(name ='Concatenate2')([L1_distance, FC_1])
    
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=['accuracy'])
    print(siamese_net.summary())

    # return the model
    return siamese_net  

def create_VGG_model_FC_End(lr=1e-4):
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()        
    pretrain_model = Model(inputs=classifer_model.input, outputs=classifer_model.get_output_at(-1))
    for layer in pretrain_model.layers: # Set all layers to be trainable
        layer.trainable = False
    for layer in pretrain_model.layers[-4:]: # last 4 layer freeze
        layer.trainable = True     
    
    model.add(pretrain_model)
    model.add(Flatten())
    model.add(Dense(5120, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    concatenation_layer = Concatenate(name ='Concatenate')([encoded_l, encoded_r])
    FC_1 = Dense(10240,  activation='relu')(concatenation_layer)
    FC_1 = Dropout(0.2)(FC_1)

    FC_1 = Dense(2560,  activation='relu')(FC_1)
    FC_1 =Dropout(0.2)(FC_1)

    FC_1 = Dense(1024,  activation='relu')(FC_1 )
      

    FC_2 = Dense(10240,  activation='relu')(L1_distance)
    FC_2 = Dropout(0.2)(FC_2)

    FC_2 = Dense(2560,  activation='relu')(FC_2)
    FC_2 =Dropout(0.2)(FC_2)

    FC_2 = Dense(1024,  activation='relu')(FC_2 )
     
    concatenation_layer2 = Concatenate(name ='Concatenate2')([FC_1, FC_2])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(concatenation_layer2)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(lr))
    print(siamese_net.summary())

    # return the model
    return siamese_net   

model = create_FC_VGG_Amr_model()
model.name = "SiameseNet_VGG16"




import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from numpy import mean
from numpy import std


keras.utils.vis_utils.pydot = pyd

#create your model
#then call the function on your model
plot_model(model, to_file='create_FC_VGG_Amr_model.png')

#To display the input and output shapes of each layer in the plotted graph
plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)



batch_size = 30
weights_path = join("create_FC_VGG_Amr_model.hd5")

checkpointer = ModelCheckpoint(weights_path, monitor="val_loss", verbose=1, mode='min', save_best_only=True)
reduceLROnPlato = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1, mode='min')
earlyStop = EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=10)

generator = Data_Generator(df, image_shape, batch_size).get_batch()
validator = Data_Generator(df, image_shape, batch_size).get_batch()


#fit_generator
#Requires two generators, one for the training data and another for validation. 
#Fortunately, both of them should return a tupple (inputs, targets) and both of them can be instance of Sequence class.

history = model.fit_generator(generator,
                              steps_per_epoch=(len(df_X_train) // batch_size) * 2,
                              validation_data=next(validator),
                              validation_steps=len(df_X_test) // batch_size,
                              epochs=200,
                              verbose=1,
                              callbacks=[checkpointer, reduceLROnPlato, earlyStop])




#score1  = model.evaluate(validator, verbose=0)
#print ("test loss--------------------",score1[0])
# Draw learning curve
def show_history(history):
    fig, ax = plt.subplots(1, figsize=(15,5))
    ax.set_title('loss')
    ax.plot(history.epoch, history.history["loss"], label="Train loss")
    ax.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax.legend()


show_history(history)


# N-way evaluation

model = load_model("create_FC_VGG_Amr_model.hd5")

class N_way_validate(object):
    
    def __init__(self, df, image_shape, n=20, k_trial=100):
        # Prepare parameters
        self.df = df.copy()
        self.h, self.w, self.c = image_shape
        self.labels = list(set(self.df['Class']))
        self.n = n
        self.k_trial = k_trial
    
    def resize_image(self, img_array):
        img = Image.fromarray(img_array)
        img = img.resize(image_shape[:-1])
        return np.array(img)

    def load_image(self, url):
        img = Image.open(url).convert('RGB')
        img = np.array(img)
        img = resize_image(img)
        img = np.divide(img, 255)
        return img

    def get_batch(self):
        
        while True:
            # Select our anchor labels
            selected_label = np.random.choice(self.labels, 1)[0]

            anchorImage_path = self.df[self.df["Class"] == selected_label].sample(n=1)['Path'].values[0]
            anchorImage = self.load_image(anchorImage_path)

            # Place holder for images
            pairs = [np.zeros((self.n, self.h, self.w, self.c)) for i in range(2)]

            # Random select the location of correct label
            targets = np.zeros((self.n,))
            targets[0] = 1
            random.shuffle(targets)

            for b in range(self.n):
                if targets[b] == 0:
                    # Negative examples
                    labels_ = self.labels.copy()
                    labels_.remove(selected_label)

                    target_label = np.random.choice(labels_, 1, replace=False)[0]
                    image2 = self.df[self.df["Class"] == target_label].sample(n=1)['Path'].values[0]
                    image2 = self.load_image(image2)
                else:
                    # positive examples
                    target_label_path = self.df[self.df["Class"] == selected_label].sample(n=1)['Path'].values[0]
                    while(target_label_path == anchorImage_path):
                        target_label_path = self.df[self.df["Class"] == selected_label].sample(n=1)['Path'].values[0]
                    image2 = self.load_image(target_label_path)

                pairs[0][b, :, :, :] = anchorImage
                pairs[1][b, :, :, :] = image2

            return pairs, targets.astype(int)
        
    def score(self, model):
        score = 0
        count = 0
        
        for k in tqdm(range(self.k_trial)):
            # Get batch
            batch, targets = self.get_batch()
            #print("***********************")
            #print(targets)
            #print("***********************")
            
            # Make prediction
            pred = model.predict(batch)
            pred = np.argmax(pred)
            #print("***********************")
            #print(pred)
            
            actual = np.argmax(targets)
            #print(actual)
            #print("***********************")
            if pred == actual:
                score += 1
            count += 1
        #print("***********************")
        #print(count)
        #print("***********************")

        precent_correct = np.round((score / self.k_trial) * 100, 3)
        print("Got an average of {} way N shot learning accuray ".format(precent_correct))        
        return precent_correct


batch, targets = N_way_validate(df, image_shape).get_batch()

for n in range(len(targets)):
    show_images([batch[0][n], batch[1][n]], mLabels[targets[n]])



# Summarize Results
# We cannot judge the skill of the model from a single evaluation. The reason for this is that
# neural networks are stochastic, meaning that a different specific model will result when training
# the same model configuration on the same data. This is a feature of the network in that it gives
# the model its adaptive ability, but requires a slightly more complicated evaluation of the model.
# We will repeat the evaluation of the model multiple times, then summarize the performance of
# the model across each of those runs. For example, we can call evaluate model() a total of 10
# times. This will result in a population of model evaluation scores that must be summarized.

# We can summarize the sample of scores by calculating and reporting the mean and standard
# deviation of the performance. The mean gives the average accuracy of the model on the dataset,
# whereas the standard deviation gives the average variance of the accuracy from the mean. The
# function summarize results() below summarizes the results of a run


scores = []
repeats = 10 

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

for r in range(repeats):
    n_validate = N_way_validate(df_X_test, image_shape, n=20)
    score = n_validate.score(model)
    #score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)
# summarize results
summarize_results(scores)