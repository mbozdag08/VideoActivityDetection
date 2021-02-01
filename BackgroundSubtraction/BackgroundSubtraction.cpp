#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <ctype.h>
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <string>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

//The kernels for the morphological operations
Mat kernel1 = getStructuringElement(MORPH_RECT, Size(2, 2)) * 255;
Mat kernel2 = getStructuringElement(MORPH_CROSS, Size(3, 3)) * 255;
Mat kernel3 = getStructuringElement(MORPH_RECT, Size(1, 5)) * 255;

void frameDifference(VideoCapture input, int width, int heigth, int frameInterval, int firstFrame, int lastFrame, String folder)
{
	//Setting the input index as the first frame number chosen by the user (-1 is for adjusting the index)
	input.set(CAP_PROP_POS_FRAMES, firstFrame - 1);

	if (!input.isOpened())
	{
		cout << "\nframeDifference Error: Video file cannot be opened. Please check the file destination." << endl;
		return;
	}

	//Declaring the variables 
	Mat frame, framePrev, frameG;
	//Creating VideoWrtier variables to write the mask and contours on avi files
	VideoWriter fgMask(folder + "/fdMask.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));
	VideoWriter original(folder + "/fdContours.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));

	Mat difference;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//Reading the first frame of the video and converting it to Grayscale
	input >> framePrev;
	cvtColor(framePrev, framePrev, COLOR_BGR2GRAY);
	//Reading the second frame
	input >> frame;


	for (int frameNum = firstFrame; frameNum <= lastFrame; frameNum = frameNum + frameInterval)
	{
		if (framePrev.empty() || frame.empty()) { break; }

		//Converting the frame to Grayscale, appliying Gaussian blur to both frames
		cvtColor(frame, frameG, COLOR_BGR2GRAY);
		GaussianBlur(frameG, frameG, Size(5, 5), 0);
		GaussianBlur(framePrev, framePrev, Size(5, 5), 0);
		//Taking the absolute difference of the frames and thresholding the difference to create the mask
		absdiff(frameG, framePrev, difference);
		threshold(difference, difference, 20, 255, 0);

		//Applying opening and dilation to create better blobs in the mask
		morphologyEx(difference, difference, MORPH_OPEN, kernel1);
		dilate(difference, difference, kernel2);
		dilate(difference, difference, kernel3);

		//Finding contours on the mask
		findContours(difference, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contoursRect(contours.size());
		vector<Rect>  boundRect(contours.size());
		vector<vector<Point> > hulls(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			//Finding convex hulls on the contours that were found to create better rectangle contours on the blobs
			convexHull(contours[i], hulls[i]);

			//Choosing hulls with size greater than 700 to filter out small rectangle contours
			if (contourArea(hulls[i]) > 700)
			{
				approxPolyDP(hulls[i], contoursRect[i], 3, true);
				boundRect[i] = boundingRect(contoursRect[i]);
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 1);
			}
			else continue;
		}

		//Converting the graycale mask to BGR to write the avi file 
		cvtColor(difference, difference, COLOR_GRAY2BGR);
		fgMask.write(difference);
		//Writing the original frame with contours on the avi file
		original.write(frame);
		//Saving the mask as individual frames to a folder as png files for comparison of methods
		String dest = folder + "/fdMask" + to_string(frameNum) + ".png";
		imwrite(dest, difference);
		//Copying the current frame to framePrev to use in the loop
		frameG.copyTo(framePrev);
		imshow("fd Original", frame);
		imshow("fd Mask", difference);

		//Skipping the unwanted frames according to the frame interval
		for (int i = 0; i < frameInterval; i++) { input >> frame; }
		if (frame.empty()) { break; }

		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}
}

void backgroundSubtraction(String method, VideoCapture input, int width, int heigth, int frameInterval, int firstFrame, int lastFrame, String folder)
{
	//Setting the input index as the first frame number chosen by the user (-1 is for adjusting the index)
	input.set(CAP_PROP_POS_FRAMES, firstFrame - 1);

	if (!input.isOpened())
	{
		cout << "\nbackgroundSubtraction - " << method << " -  Error: Video file cannot be opened. Please check the file destination." << endl;
		return;
	}

	String fileNameMask, fileNameOrg;
	//Creating a smart pointer for BackgroundSubtractor object
	Ptr<BackgroundSubtractor> BS;
	//framenum and dest is used for the saved png files of the mask
	String dest;

	//Choosing the filename and the pointer specifications depending o the input String method
	if (method == "mog2")
	{
		BS = createBackgroundSubtractorMOG2(500, 16.0, false);
		fileNameMask = folder + "/mog2Mask.avi";
		fileNameOrg = folder + "/mog2Contours.avi";

	}
	else if (method == "knn")
	{
		BS = createBackgroundSubtractorKNN(500, 400.0, false);
		fileNameMask = folder + "/knnMask.avi";
		fileNameOrg = folder + "/knnContours.avi";
	}

	//Creating videowrtier variables to write the mask and contours on avi files
	VideoWriter fgMask(fileNameMask, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));
	VideoWriter original(fileNameOrg, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));

	Mat frame, frameMask;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//Reading the first frame of the input video
	input >> frame;


	for (int frameNum = firstFrame; frameNum <= lastFrame; frameNum = frameNum + frameInterval)
	{
		if (frame.empty()) { break; }

		//Applying the specified Background Subtraction method in the input String method (mog2 or knn)
		BS->apply(frame, frameMask);

		//Applying erosion, opening, and dilation to create better blobs in the mask
		erode(frameMask, frameMask, kernel2);
		morphologyEx(frameMask, frameMask, MORPH_OPEN, kernel1);
		dilate(frameMask, frameMask, kernel2);
		dilate(frameMask, frameMask, kernel3);

		//Rest is the same with the previous method (finding contours and writing the files)
		findContours(frameMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contoursRect(contours.size());
		vector<Rect>  boundRect(contours.size());
		vector<vector<Point> > hulls(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			convexHull(contours[i], hulls[i]);

			if (contourArea(hulls[i]) > 700)
			{
				approxPolyDP(hulls[i], contoursRect[i], 3, true);
				boundRect[i] = boundingRect(contoursRect[i]);
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 1);
			}
			else continue;
		}

		cvtColor(frameMask, frameMask, COLOR_GRAY2BGR);
		original.write(frame);
		fgMask.write(frameMask);
		if (method == "mog2")
		{
			dest = folder + "/mog2Mask" + to_string(frameNum) + ".png";
		}
		else
		{
			dest = folder + "/knnMask" + to_string(frameNum) + ".png";
		}
		imwrite(dest, frameMask);
		imshow(method + " Original", frame);
		imshow(method + " Mask", frameMask);

		//Skipping the unwanted frames according to the frame interval
		for (int i = 0; i < frameInterval; i++) { input >> frame; }
		if (frame.empty()) { break; }

		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}
}

void medianSubtraction(VideoCapture input, int width, int heigth, int numOfFrames, int frameInterval, int firstFrame, int lastFrame, String folder)
{
	//Resetting the input video
	input.set(CAP_PROP_POS_FRAMES, firstFrame - 1);

	if (!input.isOpened())
	{
		cout << "\nmedianSubtraction Error 1: Video file cannot be opened. Please check the file destination." << endl;
		return;
	}

	Mat frame, subFrame, grayFrame, mask;
	//Creating a Mat type vector to store the frames
	vector<Mat> frames;

	//Choosing the first "numOfFrames" frames of the video
	for (int i = 0; i < numOfFrames; i++)
	{
		input.set(CAP_PROP_POS_FRAMES, i);
		input >> frame;
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (frame.empty())
			continue;
		//Storing the frames in the vector "frames"
		frames.push_back(frame);
	}

	//Creating an 8 bit unsigned single channel Mat variable to store the grayscale background
	Mat medianBackground(heigth, width, CV_8UC1);

	for (int row = 0; row < heigth; row++)
	{
		for (int col = 0; col < width; col++)
		{
			//Creating an integer type vector to store the pixel values
			vector<int> pixels;

			for (int imgNumber = 0; imgNumber < frames.size(); imgNumber++)
			{
				//Storing the pixel at (row,col) for all the frames that were stored in the vector "frames"
				int pixel = frames[imgNumber].at<uchar>(row, col);
				pixels.push_back(pixel);
			}
			//Taking the median of the pixels and writing it to (row,col) of the Mat variable medianBackground
			nth_element(pixels.begin(), pixels.begin() + pixels.size() / 2, pixels.end());
			medianBackground.at<uchar>(row, col) = pixels[pixels.size() / 2];
		}
	}
	imshow("Background", medianBackground);
	imwrite(folder + "/medianBackground.png", medianBackground);
	waitKey(25);
	//Applying Gaussian blur to the background frame
	GaussianBlur(medianBackground, medianBackground, Size(5, 5), 0);

	input.set(CAP_PROP_POS_FRAMES, firstFrame - 1);

	if (!input.isOpened())
	{
		cout << "\nmedianSubtraction Error 2: Video file cannot be opened. Please check the file destination." << endl;
		return;
	}

	VideoWriter medianMask(folder + "/medianMask.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));
	VideoWriter original(folder + "/medianContours.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, heigth));

	//Reading the first frame of the input video
	input >> subFrame;

	for (int frameNum = firstFrame; frameNum <= lastFrame; frameNum = frameNum + frameInterval)
	{
		if (subFrame.empty()) { break; }

		//Converting the frame to Graysclae
		cvtColor(subFrame, grayFrame, COLOR_BGR2GRAY);
		//Applying Gaussian blur to the frame
		GaussianBlur(grayFrame, grayFrame, Size(5, 5), 0);
		//Taking the absolute difference between the frame and the background
		absdiff(grayFrame, medianBackground, mask);
		//Thresholding the differenceto create the mask. Applying erosion, opening, and dilation to create better blobs in the mask
		threshold(mask, mask, 60, 255, 0);
		erode(mask, mask, kernel1);
		morphologyEx(mask, mask, MORPH_OPEN, kernel1);
		dilate(mask, mask, kernel2);
		dilate(mask, mask, kernel3);

		//Rest is the same with the previous methods (finding contours and writing the files)
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contoursRect(contours.size());
		vector<Rect>  boundRect(contours.size());
		vector<vector<Point> > hulls(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			convexHull(contours[i], hulls[i]);

			if (contourArea(hulls[i]) > 700)
			{
				approxPolyDP(hulls[i], contoursRect[i], 3, true);
				boundRect[i] = boundingRect(contoursRect[i]);
				rectangle(subFrame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 1);
			}
			else continue;
		}

		cvtColor(mask, mask, COLOR_GRAY2BGR);
		medianMask.write(mask);
		original.write(subFrame);
		String dest = folder + "/medianMask" + to_string(frameNum) + ".png";
		imwrite(dest, mask);
		imshow("median Original", subFrame);
		imshow("median Mask", mask);

		//Skipping the unwanted frames according to the frame interval
		for (int i = 0; i < frameInterval; i++) { input >> subFrame; }
		if (subFrame.empty()) { break; }

		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}
}

void compareMethod(String method, String folder, int firstFrame, int lastFrame, int width, int height, int fps, double timeInterval, int frameInterval)
{
	String methodFile, gtFile;
	Mat methodIm, gtIm;
	double eucDist = 0;
	double absDist = 0;
	//Taking the difference between the first and the last frame number to get the number of frames
	double diff = lastFrame - firstFrame;
	double frameCount = diff / frameInterval;

	for (int frame = firstFrame; frame <= lastFrame; frame = frame + frameInterval)
	{
		//Creating the names o the mask and groundtruth files
		methodFile = method + "Mask" + to_string(frame) + ".png";
		String frameNum = to_string(frame);
		int length = frameNum.length();
		gtFile = "gt" + string(6 - length, '0') + to_string(frame) + ".png";
		//Reading the files
		methodIm = imread(folder + "/" + method + "Mask" + "/" + methodFile, 0);

		if (methodIm.empty())
		{
			cout << "\n-- " << method << " -- compareMethod Error: Method image file is empty. Please check the folder destination, first frame number, and the last frame number." << endl;
			return;
		}

		gtIm = imread(folder + "/groundtruth/" + gtFile, 0);

		if (gtIm.empty())
		{
			cout << "\n-- " << method << " -- compareMethod Error: Ground truth image file is empty. Please check the folder destination, first frame number, and the last frame number." << endl;
			return;
		}

		threshold(gtIm, gtIm, 10, 255, 0);

		//Taking the euclidian distance and absolute difference between the mask and the groundtruth
		absDist = absDist + norm(methodIm, gtIm, NORM_L1);
		eucDist = eucDist + norm(methodIm, gtIm, NORM_L2);

	}
	//Taking the average difference of a frame
	double absAvg = absDist / frameCount;
	double eucAvg = eucDist / frameCount;

	cout << "\nFor " << folder
		<< "\nFPS: " << fps << "  -  Number of Frames: " << diff << "  -  Frame Interval: " << frameInterval << "  -  Time Interval: " << timeInterval << " seconds"
		<< "\nThe absolute difference between " << method << " and groundtruth is " << absDist
		<< "\nThe euclidian distance between " << method << " and groundtruth is " << eucDist << endl;

}

void compareMethod2(String method, String folder, int firstFrame, int lastFrame, int width, int height, int fps, double timeInterval, int frameInterval)
{
	String methodFile, gtFile;
	Mat methodIm, gtIm;
	int gtPix, methodPix;
	double TP = 0, TN = 0, FP = 0, FN = 0;
	double TPR, TNR, FPR, FNR, Prec;
	//Taking the difference between the first and the last frame number to get the number of frames
	double diff = lastFrame - firstFrame;
	double frameCount = diff / frameInterval;

	for (int frame = firstFrame; frame <= lastFrame; frame = frame + frameInterval)
	{
		//Creating the names o the mask and groundtruth files
		methodFile = method + "Mask" + to_string(frame) + ".png";
		String frameNum = to_string(frame);
		int length = frameNum.length();
		gtFile = "gt" + string(6 - length, '0') + to_string(frame) + ".png";
		//Reading the files
		methodIm = imread(folder + "/" + method + "Mask" + "/" + methodFile, 0);
		
		if (methodIm.empty())
		{
			cout << "\n-- " << method << " -- compareMethod2 Error: Method image file is empty. Please check the folder destination, first frame number, and the last frame number." << endl;
			return;
		}

		gtIm = imread(folder + "/groundtruth/" + gtFile, 0);

		if (gtIm.empty())
		{
			cout << "\n-- " << method << " -- compareMethod2 Error: Ground truth image file is empty. Please check the folder destination, first frame number, and the last frame number." << endl;
			return;
		}

		threshold(gtIm, gtIm, 10, 255, 0);

		//Finding the true positive, true negative, false positive and false negative values
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				int gtPix = (int)gtIm.at<uchar>(row, col);
				int methodPix = (int)methodIm.at<uchar>(row, col);

				if (gtPix == 255)
				{
					if (methodPix == 255) { TP++; }
					else { FN++; }
				}
				else
				{
					if (methodPix == 255) { FP++; }
					else { TN++; }
				}
			}
		}		
	}

	Prec = TP / (TP + FP);
	TPR = TP / (TP + FN);
	FPR = FP / (FP + TN);
	TNR = TN / (TN + FP);
	FNR = FN / (FN + TP);

	cout << "\nFor " << method << " - " << folder
		<< "\nFPS: " << fps << "  -  Number of Frames: " << frameCount << "  -  Frame Interval: " << frameInterval << "  -  Time Interval: " << timeInterval << " seconds"
		<< "\nTP: " << TP << "\nFP: " << FP << "\nTN: " << TN << "\nFN: " << FN
		<< "\n\nPrecision: " << Prec << "\nTPR (Recall): " << TPR << " \nFPR: " << FPR << "\nTNR (Specificity): " << TNR << "\nFNR: " << FNR << endl;

}

int main(int argc, char* argv)
{
	/*Creating the variables for the video file name, directory of the folder for the processed image/video files, flags for file name and folder directory inputs,
	along with the height, width, fps, and frame count of the video dile that was opened*/
	String fileName;
	String folder;
	int videoFlag = 0, folderFlag = 0;
	int width, height, fps, frameCount;

	cout << "Please enter the destination and the name of the video file. Example:"
		<< "\nFile directory: C:/Files/file.avi \nFile directory: ";
	cin >> fileName;
	//Capturing the video file
	VideoCapture cap(fileName);

	while (videoFlag == 0)
	{
		//Taking the width, height, and the fps to use in the methods
		width = cap.get(CAP_PROP_FRAME_WIDTH);
		height = cap.get(CAP_PROP_FRAME_HEIGHT);
		fps = cap.get(CAP_PROP_FPS);
		frameCount = cap.get(CAP_PROP_FRAME_COUNT);

		//If there are no frames detected, the video file is empty, so get the directory and the name of the file again
		if (frameCount == 0)
		{
			cout << "Invalid video file. Try Again. Example: \nFile directory: C:/Files/file.avi"
				 << "\nFile directory: ";
			cin >> fileName;
			cap.release();
			cap.open(fileName);
		}
		//If there are frames, then a video file is detected, so raise the flag to end the loop
		else
		{
			videoFlag = 1;
		}
	}

	//Display the width, the height, the fps, and the frame count of the detected video file
	cout << "\nWidth: " << width << "  Height: " << height << "  FPS: " << fps << "  Frame Count: " << frameCount << endl;
	
	//The flags to choose the methods to be used
	int fdFlag, mog2Flag, knnFlag, medianFlag, compareFlag, compare2Flag;
	//The number of frames that will be used in median subtraction method to create a background image
	int medianFrameNum;

	cout << "\nPlease enter the directory of the folder where you want to save the processed images."
		 << "\nWarning: The directory should contain the ground truth images in a folder named 'groundtruth'. Example:"
		 << "\nFolder directory: C:/Files \nFolder directory: ";
	cin >> folder;

	struct stat info;
	while (folderFlag == 0) 
	{
		const char* folderCh = folder.c_str();
		if (stat(folderCh, &info) == 0)
		{
			folderFlag = 1;
		}
		else
		{
			cout << "Invalid folder directory. Try Again. Example: \nFolder directory: C:/Files"
				 << "\nFolder directory: ";
			cin >> folder;
		}
	}

	cout << "\nPlease enter 0 if you don't want to use the method and 1 if you want to use it." << "\nFrame Differencing (fd): ";
	cin >> fdFlag;
	cout << "Mixture of Gaussians (mog2): ";
	cin >> mog2Flag;
	cout << "K-Nearest Neigbors (knn): ";
	cin >> knnFlag;
	cout << "Median Subtraction: ";
	cin >> medianFlag;

	if (medianFlag == 1)
	{
		cout << "Please choose the number frames to be used to construct the background for the method median subtraction: ";
		cin >> medianFrameNum;
	}

	//The interval between frames for the frame difference method and for the comparison methods
	int frameInterval;
	int firstFrame;
	int lastFrame;

	cout << "\nPlease enter the frame interval for the frame differencing method and the comparison methods (1 being consecutive frames)." << "\nFrame Interval: ";
	cin >> frameInterval;
	cout << "\nPlease enter the number of the first and the last frame that you want to use."
		<< "\n\nFirst Frame (minimum is 1): ";
	cin >> firstFrame;

	//Equating the first frame to 1, since that is the minimum index for the files 
	if (firstFrame <= 0) firstFrame = 1;
	
	cout << "Last Frame (maximum is " << frameCount - 1  << "): ";
	cin >> lastFrame;

	//Equating the last frame to the frame count minus 1, since that is the maximum number of frames that the frame differencing method can create 
	if (lastFrame >= frameCount) lastFrame = frameCount - 1;

	//The time interval between each frame in seconds, calculated by using the frame interval and the fps
	double timeInterval = (double)(frameInterval) / (double)fps;

	cout << "\nFPS: "<< fps <<" Frame Interval: " << frameInterval << " Time Interval: " << timeInterval << endl;

	//Calling the methods
	if (fdFlag == 1) 
	{ 
		String fdFolder = folder + "/fdMask";
		const char* fdFolderCh = fdFolder.c_str();
		_mkdir(fdFolderCh);
		frameDifference(cap, width, height, frameInterval, firstFrame, lastFrame, fdFolder);
	}
	if (mog2Flag == 1)
	{ 
		String mog2Folder = folder + "/mog2Mask";
		const char* mog2FolderCh = mog2Folder.c_str();
		_mkdir(mog2FolderCh);
		backgroundSubtraction("mog2", cap, width, height, frameInterval, firstFrame, lastFrame, mog2Folder);
	}
	if (knnFlag == 1) 
	{ 
		String knnFolder = folder + "/knnMask";
		const char* knnFolderCh = knnFolder.c_str();
		_mkdir(knnFolderCh);
		backgroundSubtraction("knn", cap, width, height, frameInterval, firstFrame, lastFrame, knnFolder);
	}
	if (medianFlag == 1)
	{ 
		String medianFolder = folder + "/medianMask";
		const char* medianFolderCh = medianFolder.c_str();
		_mkdir(medianFolderCh);
		medianSubtraction(cap, width, height, medianFrameNum, frameInterval, firstFrame, lastFrame, medianFolder);
	}

	//First and last frames that will be used in the comparison methods
	int firstCompFrame, lastCompFrame;

	cout << "\nPlease enter 0 if you don't want to use the comparison method and 1 if you want to use it." 
		 << "\nAbsolute Difference and Euclidian Distance Comparison: ";
	cin >> compareFlag;
	cout << "Binary Classification Comparison: ";
	cin >> compare2Flag;

	if (compareFlag == 1 || compare2Flag == 1) {

		cout << "\nPlease enter the number of the first and the last frame for the comparison methods."
			<< "\n\nWarning: Consider the frame interval, the first frame, and the last frame numbers."
			<< "\nThe numbers should match for the program to work. Otherwise, it will return with an"
			<< "\nerror. Check the ground truth files and choose the first frame number accordingly,"
			<< "\nsince the ground truth files might contain empty frames in the beginning."
			<< "\n\nFirst Frame for Comparison (minimum is " << firstFrame + (5 * frameInterval) << "): ";
		cin >> firstCompFrame;

		//Equating the first frame of comparison to the first frame of methods plus 5 times the frame interval, due to knn and mog2 using first 4-5 frames
		if (firstCompFrame < (firstFrame + 5 * frameInterval)) firstCompFrame = firstFrame + 5 * frameInterval;

		cout << "Last Frame for Comparison (maximum is " << lastFrame << "): ";
		cin >> lastCompFrame;

		//Equating the last frame of comparison to the last frame of methods
		if (lastCompFrame > lastFrame) lastCompFrame = lastFrame;

		//Comparing the methods with euclidian distance and absolute difference
		if (compareFlag == 1)
		{
			compareMethod("fd", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod("mog2", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod("knn", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod("median", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
		}

		//Comparing the methods with pixelwise binary classification
		if (compare2Flag == 1)
		{
			compareMethod2("fd", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod2("mog2", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod2("knn", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
			compareMethod2("median", folder, firstCompFrame, lastCompFrame, width, height, fps, timeInterval, frameInterval); 
		}
	}

	destroyAllWindows();
	return 0;
}
