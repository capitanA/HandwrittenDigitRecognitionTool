CONTENTS OF THIS FILE
---------------------

 * [INTRODUCTION](#Introduction)
 * [REQUIREMENTS](#Requirements)
 * [INSTALLATION](#Installation)
 * [GUI](#Gui)
 * [TEST](#Test)
 * [CONFIGURATION](#Configuration)
 * [TROUBLESHOOTING](#Troubleshooting)
 * [SUPPORT](#Support)



Introduction
------------
The purpose of this project is to recognize handwritten digits inside
enclosed shapes.

I used the famous MNIST dataset for training a CNN
model.The MNIST database of handwritten digits was used to train the model.
This dataset has a training set of 60,000 examples, and a test set of 10,000
examples. It is a subset of a larger set available from NIST. 
Each image is of a dimension, 28×28 i.e. total 784 pixel values.


The project file consists of two scripts:
* **train.py:** this script trains the model based on MNIST dataset to 
  recognize a handwritten digit. CNN algorithm was used to train the model,
  and it's accuracy is 99.26%. (the callback module was embedded to stop the training once it hits the %99 accuracy.)
  After training, the model was saved in **mnist.h5** file. The Accuracy of the algorithm depends on the preprocessing step for extracting the digits. In **DigitRecognition.py** script, preprocessing techniques from opencv
  has been used to extract a clear digit out of the input image.
  So many factors need to be considered in training process to achieve an accurate model such as number of layer, epoch, batch size, active functions, error functions etc. The epoch was set to 12 as it had the best result by trial and error.
* **DigitRecognition.py:** in this script, the mnist.hs model was loaded
  and used to recognize digits. Using **connected component** and
  **findContour** algorithms the enclosed shapes was detected and numbers 
  outside them were filtered out, then the digit was detected one by one using the model and their locations was stored in **user_logger.log** file. 
  
Requirements
------------
All the required library and framework listed bellow:
* OpenCV
* Numpy
* Tensorflow
* Keras
* pillow

Note that there is a requirement.txt file in the project folder that you can install them from there.



Installation
------------
1- First it is suggested to create a virtual environment for the project:

    python3 -m venv venv

2- Install the requirements.txt file:

    pip install -r requirements.txt

3- In a case you want to train the model again: 

    python train.py

4- Run the project:

    python digit_recognition.py

Gui
-------------
I designed a very basic gui using tkinter just for your convenience.
There is a button for uploading an input image. Everytime when you want to test another image you just need to hit the reset button before uploading.


Test
-------------
I have tested 10 different handwritten images on this model. in most cases the model prediction was satisfying while in some more preprocessing techniques is required. 
Test images are available in images folder and the outputs images were saved in output folder.
There is a user_logger.log file at this directory as well in which all the coordinates was stored.


Configuration
-------------

There is no configuration

Troubleshooting
-----------

If the resulted image is not the one you desired, there may be a possibility
of shadows in the input image. For images with shadow, uncomment the 
**shadow_removal** method in the **DigitRecognition.py** script (line 132).
This method eliminates the shadow caused by a camera when taking a picture
from handwritten document.


Support
-----------

If you are having issues, please let us know.
* Arash Fassihozzaman Langroudi - email: afassihozzam@mun.ca

