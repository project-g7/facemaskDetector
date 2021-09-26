# from itertools import Predicate
import numpy as np  # for array functions
import os # to access data set
import matplotlib.pyplot as plt # to create graphs
from imutils import paths # to access data set 
# import CNN Structures 
# tensorflow/ keras are deep learning libraries
# mobile net is a architecture. can use with mobiles and rasberyPy 
# monileNetV2 used to traininga the model
from tensorflow.keras.applications import MobileNetV2 
# 
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
# gradient dicent function
from tensorflow.keras.optimizers import Adam
# before use need to pre-process the data 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# need to pre-process the images also. generate more number of data set 
# using less data sets.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Model can not take images in photo form therefore images need to convert in to array
from tensorflow.keras.preprocessing.image import img_to_array
# before images convert in to array they need to load 
from tensorflow.keras.preprocessing.image import load_img
# to categorize 
from tensorflow.keras.utils import to_categorical
# sklearn is machine learning library 
# we have two categories Mask and No mask (labels). 
# model need them as inputs so those label need to convert in to binary form.  
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# for classification report 
from sklearn.metrics import classification_report
from tensorflow.python.framework.tensor_util import _TENSOR_CONTENT_TYPES



# access data set. r for reverse ?
dataset = r'D:\Projects\Face Mask Detector_3-Final_3\dataset'
# abouve imported paths, there is function list_images 
# that create a list of images from dataset
imagePaths = list(paths.list_images(dataset))
# print(imagePaths)

# x - data(images) y - lables - with mask or without mask 
data=[]
labels = []

# this is one image path 
# D:\\Projects\\Face Mask Detector_3-Final_2\\dataset/without_mask\\399.jpg'
# here -2 direct to label in image path. [-1] = 399.jpg [-2] = without_mask
for i in imagePaths:
    label = i.split(os.path.sep)[-2]
    # images got different sizes so set all in to 224x224
    image = load_img(i, target_size=(224,224))
    # convert images to array. cannot input pictures to model
    # [249. 238. 234.]
    #[246. 235. 229.]
    #[235. 225. 216.]] part of array 
    image = img_to_array(image) # downdraded pillow version from  8.3 to 8.2
    # after pre-processed numbers in the arra
    # y are become scale down 
    #[ 0.6784314   0.62352943  0.48235297]
    #[ 0.6627451   0.62352943  0.47450984]
    #[ 0.654902    0.6156863   0.4666667 ] 
    image = preprocess_input(image)
    # each label of images append to lables array
    labels.append(label)
    # each image append to the data array
    data.append(image)

# now data and labels are in list. to train the model need them convert in to numpy array
#[-0.94509804 -0.9764706  -0.85882354]
#[-0.94509804 -0.9764706  -0.85882354]
#[-0.94509804 -0.9764706  -0.85882354]
data = np.array(data, dtype='float32')
labels = np.array(labels)

# call LabelBinarize and storing in to lb
lb = LabelBinarizer()
# read the labels convert them to binary and categorized 
# fit_transform - Fit label binarizer and transform multi-class labels to binary labels.
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# training data = 80% / testing data = 20%
# split data in to traininga and testing 
# random state - if random state is not a integer the train and test data splited randomly 
# stratify - training and testing data also splited in the same percentages as with mask and with out mask
# if with mask = 60% and without mask = 40%, in training set also got 60% of with mask and 40% of without mask
trainX, testX, trainY, testY = train_test_split(data, labels, test_size= 0.20, random_state=10, stratify=labels)
# 80% of whole images = 1100, 224x224 = size of images, 3 = RGB image 
print(trainX.shape) # output - (1100, 224, 224, 3)
# 80% of whole images = 1100, 2 = categories(0,1)
print(trainY.shape ) # output = (1100, 2)
# 20% of whole images = 276, 224x224 = size of images, 3 = RGB image 
print(testX.shape) # output - (276, 224, 224, 3)
# 20% of whole images = 276, 2 = categories(0,1)
print(testY.shape ) # output = (276, 2)

# generating larger set of data using given data set 
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')

# CNN = Convolutional NeuralNetwork 
# pre tarined model/CNN Architecture - MobileNetV2, in this code mobileNetv2 model is downloading 
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

# summary of the base model
# baseModel.summary()
# 
headModel = baseModel.output
# take avarage features extra - maxPooling take tha maximum featuer 
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# flattern the layers
headModel = Flatten(name="flatten")(headModel)
# connecting 128 neurons of layer - dense layer 
headModel = Dense(128, activation="relu")(headModel)
# Dropout layer prevent your model from overfitting 
headModel = Dropout(0.5)(headModel)
# softMax - 
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

learning_rate = 0.001
# how many times to model to go through images 
Epochs = 30
# how many images you want to train your model advance BS -batch size  
BS = 20
# opt - optimizer 
# compile our model
opt = Adam(lr=learning_rate, decay=learning_rate / Epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the head of the network
model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	# steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=Epochs)

# save trained model
model.save(r'D:\Projects\Face Mask Detector_3-Final_3\mobileNet_v2.model')

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
# argmax - maximum argument target_names = 10 and 01
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))








print("For Now you are safe!!!")



