/*
 * Benchmark.cpp
 *
 *  Created on: Jun 3, 2014
 *      Author: dailos
 */

#include "Benchmark.h"
#include "opencv2/opencv.hpp"
#include "FITS.h"
#include "OpticalSystem.h"
#include "Zernikes.h"
#include "PDTools.h"
#include "AWMLE.h"
#include "WaveletTransform.h"

void process_AWMLE()
{

  cv::Mat dat, psf_data, img, psf0, psf;
  readFITS("/home/dailos/PDPairs/12-06-2013/pd.004.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);
  std::cout << img.cols << "x" << img.rows << std::endl;
  int X(0),Y(0);
  //int X(110),Y(110);

  cv::Rect rect1(X, Y, 936, 936);
  //cv::Rect rect1(X, Y, 256, 256);

  cv::Mat d0 = img(rect1).clone();

  readFITS("/home/dailos/workspace/psf_test.fits", psf);

  cv::Mat estimatedObject, estimatedObject_norm;

  double sigmaNoise = 0.00584497;
  AWMLE(d0, psf, estimatedObject, sigmaNoise, 4);
  std::cout << "d0.at<double>(30,30): " << d0.at<double>(30,30) << std::endl;
  std::cout << "estimatedObject.at<double>(30,30): " << estimatedObject.at<double>(30,30) << std::endl;

  std::cout << "d0.at<double>(40,40): " << d0.at<double>(40,40) << std::endl;
  std::cout << "estimatedObject.at<double>(40,40): " << estimatedObject.at<double>(40,40) << std::endl;

  std::cout << "d0.at<double>(50,50): " << d0.at<double>(50,50) << std::endl;
  std::cout << "estimatedObject.at<double>(50,50): " << estimatedObject.at<double>(50,50) << std::endl;

  std::cout << "d0.at<double>(60,60): " << d0.at<double>(60,60) << std::endl;
  std::cout << "estimatedObject.at<double>(60,60): " << estimatedObject.at<double>(60,60) << std::endl;
  cv::imshow("d0",d0);
  cv::imshow("psf0",psf);
  cv::imshow("estimatedObject",estimatedObject);
  cv::waitKey();
}

void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow, const int& sideLength, const double& apodizedAreaPercent, int datatype)
{
  int apodizedArea = int((apodizedAreaPercent * sideLength) / 100);
  std::cout << "apodizedArea: " << apodizedArea << std::endl;
  cv::Mat hann;
  cv::createHanningWindow(hann, cv::Size(apodizedArea * 2, 3), datatype);
  cv::Mat modifiedHanningSlice = cv::Mat::ones(1,sideLength, datatype);
  (hann(cv::Rect(0,1,apodizedArea,1))).copyTo(modifiedHanningSlice(cv::Rect(0,0,apodizedArea,1)));
  (hann(cv::Rect(apodizedArea,1,apodizedArea,1))).copyTo(modifiedHanningSlice(cv::Rect((modifiedHanningSlice.cols-apodizedArea),0,apodizedArea,1)));
  //Matrix multiplications of one single colum 1xN by one single row Nx1 matrices, to create a NxN
  modifiedHanningWindow = modifiedHanningSlice.t() * modifiedHanningSlice;
}
