![Untitled](https://user-images.githubusercontent.com/102634674/177964315-7cd28ae4-423f-4273-8f09-4cb9467f2391.png)

# real-time-facial-emotion-detection

you can download the data from https://drive.google.com/file/d/1fSF4L4R7XluCaemVCfbzyGca49hL6MRe/view?usp=sharing

This repository contains the codes, and data for real time face emotion detection from webcam.

For face dection, the cv2.haarcascade face detection model is utilized

Two approaches are considered for emotion detection: 
1- training a CNN from scratch
2- using FER library

In order to train your model,  run train.py and then uncomment line 16 in facial_emotion_detection.py
* A pretrained model in saved in dir/models

here is the video of final work:
https://user-images.githubusercontent.com/102634674/177962407-f5a9d2c2-5b61-4baf-ad9a-67f05885fc41.mp4

