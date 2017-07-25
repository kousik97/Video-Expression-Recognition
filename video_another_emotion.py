from __future__ import print_function
import argparse
import cv2
import os 
import PIL
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
import numpy as np
import cPickle 
import numpy
import cv2
import scipy
import csv
import imutils
from skimage import io
import dlib
import json
import time
import pandas as pd 
img_rows, img_cols = 48, 48
# the CIFAR10 images are RGB
img_channels = 1

model = Sequential()
model.add(Convolution2D(64, 5, 5, border_mode='valid',
						input_shape=(img_rows, img_cols,1)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
  
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
 
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
 
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
 
  
model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
 
  
model.add(Dense(7))
  
  
model.add(Activation('softmax'))
  
  
ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
			  optimizer=ada,
			  metrics=['accuracy'])


filepath="./Model.120-0.6343.hdf5"
print(filepath)
model.load_weights(filepath)

def Flip(data):
	dataFlipped = data[..., ::-1].reshape(2304).tolist()
	return dataFlipped
def Roated15Left(data):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 20, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(2304).tolist()
def Roated15Right(data):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -20, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(2304).tolist()

def StretchedVertical(data):
	datastretchedvertical = scipy.misc.imresize(data,(48,80))
	datastretchedvertical = datastretchedvertical[:,17:65]
	datastretchedvertical = datastretchedvertical.reshape(2304).tolist()
	return datastretchedvertical
def StretchedHorizontal(data):
	datastretchedhorizontal = scipy.misc.imresize(data,(60,48))
	datastretchedhorizontal = datastretchedhorizontal[5:53,:]
	datastretchedhorizontal = datastretchedhorizontal.reshape(2304).tolist()
	return datastretchedhorizontal
def Zoomed(data):
	datazoomed = scipy.misc.imresize(data,(60,60))
	datazoomed = datazoomed[5:53,5:53]
	datazoomed = datazoomed.reshape(2304).tolist()
	return datazoomed

def shiftedUp20(data):
	translated = imutils.translate(data, 0, -5)
	translated2 = translated.reshape(2304).tolist()
	return translated2
def shiftedDown20(data):
	translated = imutils.translate(data, 0, 5)
	translated2 = translated.reshape(2304).tolist()
	return translated2

def shiftedLeft20(data):
	translated = imutils.translate(data, -5, 0)
	translated2 = translated.reshape(2304).tolist()
	return translated2
def shiftedRight20(data):
	translated = imutils.translate(data, 5, 0)
	translated2 = translated.reshape(2304).tolist()
	return translated2

def flatten_matrix(matrix):
	vector = matrix.flatten(1)
	vector = vector.reshape(1, len(vector))
	return vector
def zca_whitening(inputs):
	sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
	U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
	epsilon = 0.1                #Whitening constant, it prevents division by zero
	ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
	return np.dot(ZCAMatrix, inputs)   #Data whitening
def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
							  sqrt_bias=10, min_divisor=1e-8):

	"""
	__author__ = "David Warde-Farley"
	__copyright__ = "Copyright 2012, Universite de Montreal"
	__credits__ = ["David Warde-Farley"]
	__license__ = "3-clause BSD"
	__email__ = "warde?far@iro"
	__maintainer__ = "David Warde-Farley"
	.. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
	   Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
	   http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
	"""
	assert X.ndim == 2, "X.ndim must be 2"
	scale = float(scale)
	assert scale >= min_divisor
	mean = X.mean(axis=1)
	if subtract_mean:
		X = X - mean[:, numpy.newaxis]  # Makes a copy.
	else:
		X = X.copy()

	if use_std:
		ddof = 1
		if X.shape[1] == 1:
			ddof = 0

		normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
	else:
		normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

	# Don't normalize by anything too small.
	normalizers[normalizers < min_divisor] = 1.

	X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
	return X
def ZeroCenter(data):
	data = data - numpy.mean(data,axis=0)
	return data

def normalize(arr):
	for i in range(3):
		minval = arr[...,i].min()
		maxval = arr[...,i].max()
		if minval != maxval:
			arr[...,i] -= minval
			arr[...,i] *= (255.0/(maxval-minval))
	return arr



def convertPercentTage(Array):
	result = []
	Sum = float(0)
	for x in range(0,7):
		Sum += float(Array[0][x])
	for i in range(0,7): 
		result.append(float(Array[0][i])/Sum)
	return result


def ConvertToArrayandReshape(List):
	numpyarray = numpy.asarray(List)
	numpyarray = numpyarray.reshape(1,48,48)
	numpyarray = numpyarray.reshape(1, 48, 48,1)
	numpyarray = numpyarray.astype('float32')
	return numpyarray
def imagePreprocessing(crop2):
	data2 = ZeroCenter(crop2)
	data3 = zca_whitening(flatten_matrix(data2)).reshape(48,48)
	data4 = global_contrast_normalize(data3)
	data5 = numpy.rot90(data4,3)
	return data5
def predict(data5):
	Train_x_Init = ConvertToArrayandReshape(data5)
	Train_x_Flip  = ConvertToArrayandReshape(Flip(data5))
	Train_x_Rotleft = ConvertToArrayandReshape(Roated15Left(data5))
	Train_x_Rotright = ConvertToArrayandReshape(Roated15Right(data5))
	Train_x_ShiftedUp = ConvertToArrayandReshape(shiftedUp20(data5))
	Train_x_ShiftedDown = ConvertToArrayandReshape(shiftedDown20(data5))
	Train_x_ShiftedLeft = ConvertToArrayandReshape(shiftedLeft20(data5))
	Train_x_ShiftedRight = ConvertToArrayandReshape(shiftedRight20(data5))
	a= np.array([  model.predict_proba(Train_x_Init, verbose=1)[0]])
	'''a = np.array([  model.predict_proba(Train_x_Init, verbose=1)[0], 
			model.predict_proba(Train_x_Flip, verbose=1)[0], 
			model.predict_proba(Train_x_Rotleft, verbose=1)[0], 
			model.predict_proba(Train_x_Rotright, verbose=1)[0],  
			model.predict_proba(Train_x_ShiftedUp, verbose=1)[0], 
			model.predict_proba(Train_x_ShiftedDown, verbose=1)[0], 
			model.predict_proba(Train_x_ShiftedLeft, verbose=1)[0], 
			model.predict_proba(Train_x_ShiftedRight, verbose=1)[0],  
		])'''
	return a.mean(axis=0)
def loadData(imagepath):
	img = io.imread(imagepath)
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	detector = dlib.get_frontal_face_detector()
	grayimg = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
	start_time = time.time()
        #print("flag3")
	dets, scores, idx= detector.run(grayimg,1)
        #print("flag4")
	#print("- %s seconds sdssd---" % (time.time() - start_time))
	start_time = time.time()
	imgWithRect = cv2.imread(imagepath)
	ListOfFaces = []
	FaceWriteList = []
        temp=0
	for rectangle in dets:
                if(scores[temp]>0.7):
		  left = rectangle.left()
		  top = rectangle.top()
		  right = rectangle.right()
		  bottom = rectangle.bottom()
		  if top<0:
		       	offset = 0-top
			top = 0
			bottom = bottom+offset
		  if left<0:
			offset = 0 -left
			left =0
			right = right+offset
		  cv2.rectangle(imgWithRect,(left,top),(right,bottom),(255,255,255),2)
		  crop = grayimg[top:bottom, left:right]
                  colored_crop=img[top:bottom, left:right]
		  crop2 = cv2.resize(crop, (48, 48)) 
		  data = imagePreprocessing(crop2)
		  ListOfFaces.append(data)
		  FaceWriteList.append(colored_crop)
                temp+=1
	#cv2.imwrite(imagepath,imgWithRect)

	#print("--- %s seconds ---" % (time.time() - start_time))

	return ListOfFaces, FaceWriteList
	
def main(pathpic):
        #print("flag1")
        start=time.time()
	ListOfFaces,FaceWriteList = loadData(pathpic)
        print("Loading Data took:",time.time()-start)
        #print("flag2")
        temp=1
        ToReturnList=[]
        os.chdir("Extracted_faces")
	for face, coors in zip(ListOfFaces, FaceWriteList):
                start=time.time()
		result = predict(face)
                print("Time taken:", time.time()-start)
                path= ((pathpic.split('/')[-1]).split('.')[0])+"-"+str(temp)+".jpg" 
                cv2.imwrite(path,coors)
                temp+=1
                ToReturnList.append([path,result])
        os.chdir("../")
        return(ToReturnList) 



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to input directory of images")
    args=vars(ap.parse_args())
    if args["video"] and os.path.isfile(args["video"]):
       df =pd.DataFrame(columns=["Video_name","frame_name","face_name","Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
       os.system("mkdir Results")
       os.chdir("Results")
       os.system("mkdir Extracted_frames")
       os.chdir("Extracted_frames")
       video_name=((args["video"].split('/')[-1]).split('.')[0])
       path= video_name+"-" +"%4d.jpg"
       command="avconv -i "+str(args["video"])+" -r 2 "+path
       os.system(command)
       images=os.listdir('.')
       print(images)
       os.system("mkdir Extracted_faces")
       for x in images:
             temp_list=main(x)
             for y in temp_list:
                list_to_append=[video_name,x,y[0],y[1][0],y[1][1],y[1][2],y[1][3],y[1][4],y[1][5],y[1][6]]
                df=df.append(pd.Series(list_to_append,index=["Video_name","frame_name","face_name","Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]),ignore_index=True)
       name="result"+".csv"
       df.to_csv(name, sep=',')
       print("Sucess, Completed")         
		     
       
