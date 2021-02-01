# Video-Activity-Detection-with-Background-Subtraction
A Video Activity Detection program that uses four different Background Subtraction techniques: Frame Differencing, Mixture of Gaussians, KNN, Median Subtraction./
In my implementation, I used four different methods to detect activity in different environments. The video samples that I used include the surveillance footage of an office, a subway station, a park, and a highway. I chose these samples because they represent a variety of environments that can be used to compare the efficiency of the algorithms and see how they perform on different types of scenarios. I used VisualStudio IDE in Windows 10 Home 64-bit and Eclipse for C++ on Ubuntu 20.04.1 64-bit, together with the OpenCV library to implement all of my methods on C++./
All of my methods have similar parameters, taken as inputs from the user. Each method starts similarly to the other ones. First, I set the video file, which is a “VideoCapture” object of the OpenCV library, to the first frame number determined by the user minus 1 frame. The reason I do this is that after I use the “VideoCapture” object, the index of the frame stays where the last method left, so I need to reset the number. I subtract 1 frame from the first frame index determined by the user because the ground truth files that will be used for comparison are indexed starting from 1, while the “VideoCapture” object indexes the frames starting from 0. After I adjust the frame index, I declare the variables that are needed in the methods. The rest of the operations include similarities, but since most of them are specific to the methods, they are explained separately.
