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
  cv::Mat WavefrontSensing(const std::vector<cv::Mat>& d, const std::vector<double>& meanPowerNoise);
  void ista(cv::Mat& u,  const std::function<double(cv::Mat)>& func, const std::function<cv::Mat(cv::Mat)>& grad);
  
private:
  std::vector<double> diversityFactor_;

};
#endif /* WAVEFRONTSENSOR_H_ */

