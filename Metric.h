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
  Metric();
  virtual ~Metric();
  //Getters
  cv::Mat Q()const {return Q_;};
  cv::Mat F()const {return F_;};
  double L()const {return L_;};
  
  void characterizeOpticalSystem(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, std::vector<OpticalSystem>& OS);
  void computeQ(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& Q);
  void computeAccSjDj(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase,
                      const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& accSjDj);
  double objectiveFunction(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  void objectEstimate(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  void noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, cv::Mat& filter);
  cv::Mat finiteDifferencesGradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  cv::Mat gradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  
private:
  cv::Mat F_;   //Object estimate
  double  L_;   //objective function evaluation at point
  cv::Mat g_;   //gradient vector at point "coeffs"
  cv::Mat coeffs_;  //Coefficients ue
  cv::Mat H_;   //noise filter, low-pass scharmer filter
  
  //Intermal metrics
  cv::Mat Q_;
  cv::Mat accSjDj_;
  std::vector<OpticalSystem> OS_;

};

#endif /* METRIC_H_ */

