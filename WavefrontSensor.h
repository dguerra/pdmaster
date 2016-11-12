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
  cv::Mat WavefrontSensing(const std::vector<cv::Mat>& d, const double& meanPowerNoise);
  static void subimageLayout(const cv::Size& dim, const unsigned int& pixelsBetweenTiles, const unsigned int& tileSize, std::vector<std::pair<cv::Range,cv::Range> >& tileRngs);
  
private:
  std::vector<double> diversityFactor_;

};
#endif /* WAVEFRONTSENSOR_H_ */

