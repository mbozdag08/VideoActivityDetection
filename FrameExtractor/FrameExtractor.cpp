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

using namespace cv;
using namespace std;

int main(int argc, char* argv)
{
	{
		VideoCapture cap("fDiffContours.avi");

		int frameNum = 0;
		Mat frame;

		if (!cap.isOpened())
		{
			cout << "Error opening video file" << endl;
		}

		while (cap.isOpened())
		{
			cap >> frame;
			if (frame.empty()) { break; }
			String dest = "C:/Users/musta/source/repos/BackgroundSubtraction/FrameExtractor/frames/fdContours" + to_string(frameNum) + ".png";
			imwrite(dest, frame);
			frameNum++;
			char c = (char)waitKey(25);
			if (c == 27)
				break;
		}
	}
}
