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

//#include "ErrorMetric.h"
//member elements: number of iterations, zernike used in the process...

class WavefrontSensor
{
public:
  WavefrontSensor();
  virtual ~WavefrontSensor();
  void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow,
      const int& sideLength, const double& apodizedAreaPercent, int datatype);
  cv::Mat WavefrontSensing(const std::vector<cv::Mat>& d, const std::vector<double>& meanPowerNoise);
private:
  //void showRestore(ErrorMetric errMet, cv::Mat& fm);

  unsigned int maximumIterations_;
  cv::Mat c_InitialValues;    //Initial values for zernike coefficients, they are zero by default

  cv::Mat cMinimum_;
  double lmMinimum_;
  unsigned int iterationMinimum_;
  std::vector<double> diversityFactor_;
  //cv::Mat zernikesInUse_;

};
#endif /* WAVEFRONTSENSOR_H_ */

