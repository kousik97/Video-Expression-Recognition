# Video-Expression-Recognition
Python script to recognise emotions of all facial images in a video.
Given a video, the script will recognise all facial images and each face is associated with an expression(anger, disgust, fear, sadness, surprise, joy and neutral).

## Input-Output 
### Input
Video with/without faces
### Output 
For a chosen unique cluster(face), graph of how its facial expression/emotion is changing through time.

## Requirements 
* Keras=1.2.2
* Tensorflow=1.2.1
* Dlib=19.4.99
* OpenCV

The codes are tested in container built from Ubuntu 14.04 CPU docker image downloaded from floyd-hub(link given below).

## Demo 

1. Download necessary weight files from [Weight-files](https://drive.google.com/open?id=0ByDWS1KXv3sodERVQXVraUc0NkU)
2. Call Demo.py by running "python demo.py -v /path/to/video" 
3. Follow runtime instructions 


