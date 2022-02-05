#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  Mat im = imread("test.jpg", 1);
  if (im.empty())
  {
    cout << "Cannot open image!" << endl;
    return -1;
  }

  imshow("image", im);
  waitKey(0);

  return 0;
}