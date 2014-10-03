/*
 * WavefrontSensing.h
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#ifndef WAVEFRONTSENSOR_H_
#define WAVEFRONTSENSOR_H_
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

#include "ErrorMetric.h"
//member elements: number of iterations, zernike used in the process...

class WavefrontSensor
{
public:
  WavefrontSensor();
  virtual ~WavefrontSensor();
  void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow,
      const int& sideLength, const double& apodizedAreaPercent, int datatype);
  cv::Mat WavefrontSensing(const cv::Mat& d0, const cv::Mat& dk, const double& meanPowerNoiseD0, const double& meanPowerNoiseDk);
private:
  void showRestore(ErrorMetric errMet, cv::Mat& fm);
  cv::Mat backToImageSpace(const cv::Mat& fourierSpaceMatrix, const cv::Size& centralROI);
  unsigned int imageCoreSize_;

  double dcRMS_Minimum_;
  double lmIncrement_Minimum_;
  unsigned int maximumIterations_;
  cv::Mat c_InitialValues;    //Initial values for zernike coefficients, they are zero by default

  cv::Mat cMinimum_;
  double lmMinimum_;
  unsigned int iterationMinimum_;
  double diversityFactor_;
  //cv::Mat zernikesInUse_;

};
#endif /* WAVEFRONTSENSOR_H_ */

