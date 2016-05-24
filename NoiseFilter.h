/*
 * NoiseFilter.h
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */

#ifndef NOISEFILTER_H_
#define NOISEFILTER_H_

#include <iostream>
#include "Optics.h"

class NoiseFilter
{
public:
  NoiseFilter(const cv::Mat& T0, const cv::Mat& Tk, const cv::Mat& D0, const cv::Mat& Dk,
              const cv::Mat& Q2, const double& sigma2NoiseD0, const double& sigma2NoiseDk);
  NoiseFilter();
  virtual ~NoiseFilter();
  cv::Mat H(){return H_;};
private:
  cv::Mat H_;  //filter noise
};
#endif /* NOISEFILTER_H_ */

