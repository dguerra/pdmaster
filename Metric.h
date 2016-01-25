/*
 * NoiseFilter.h
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */

#ifndef METRIC_H_
#define METRIC_H_

#include <iostream>
#include "OpticalSystem.h"
#include "opencv2/opencv.hpp"

class Metric
{
public:
  Metric(const std::vector<cv::Mat>& D, const double& pupilRadiousPxls);
  virtual ~Metric();
  //Getters
  cv::Mat F() const {return F_;};
  
  void characterizeOpticalSystem(const cv::Mat& coeffs, std::vector<OpticalSystem>& OS);
  void computeQ(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& Q);
  void computeP(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& P);
  void compute_dQ(const cv::Mat& zernikeElement, const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dQ);
  void compute_dP(const cv::Mat& zernikeElement, const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dP);
  
  void compute_dSj(const OpticalSystem& osj, const cv::Mat& zernikeElement, cv::Mat& dSj);

  double objective(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise);
  cv::Mat gradient(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise);
  //Compute Jacobian
  void phi(      const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, std::vector<cv::Mat>& De );
  void jacobian( const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, std::vector<std::vector<cv::Mat> >& jacob );
  void noiseFilter(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter);

private:
  cv::Mat F_;   //Object estimate
  std::vector<cv::Mat> zernikeBase_;
  double pupilRadiousPxls_;
  std::vector<cv::Mat> D_;   //data images
};

#endif /* METRIC_H_ */