# Video-Expression-Recognition
Python script to recognise emotion of all facial images in a video.
Given a video, the script will recognise all facial images and associate each face with an expression(anger, disgust, fear, sadness, surprise, joy and neutral).

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
* ffmpeg

The code is tested in a container built from Ubuntu 14.04 CPU docker image downloaded from floyd-hub(link given below).

## Demo 

1. Download necessary weight files from [weight-files](https://drive.google.com/open?id=0ByDWS1KXv3sodERVQXVraUc0NkU)
2. Extract all files downloaded from the above directory to the directory containing demo.py
3. Call demo.py by running "python demo.py -v /full/path/to/video" 
4. Follow runtime instructions 

For a quick testing, use the video provided in example_videos directory.

## References
* [Docker Image](https://github.com/floydhub/dl-docker)
* Several ideas from [ICMI2015-ChaZhang](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icmi2015_ChaZhang.pdf) are used to create the expression recognition model.
* This code is built on top of [Facial-Expression-Recognition](https://github.com/LamUong/FacialExpressionRecognition)
