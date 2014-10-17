/*
 * NOISEESTIMATOR.h
 *
 *  Created on: Oct 16, 2013
 *      Author: dailos
 */

#ifndef NOISEESTIMATOR_H_
#define NOISEESTIMATOR_H_
#include "opencv2/opencv.hpp"

class NoiseEstimator
{
public:
  NoiseEstimator();
  virtual ~NoiseEstimator();
  void kSigmaClipping(const cv::Mat& img);
  void meanPowerSpectrum(const cv::Mat& img);
  double sigma()const {return sigma_;};
  double meanPower()const {return meanPower_;};
  double sigma2()const{return sigma2_;};
private:
  double sigma2_;
  double sigma_;
  double meanPower_;
};

#endif /* NOISEESTIMATOR_H_ */
