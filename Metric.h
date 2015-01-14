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
  cv::Mat F()const {return F_;};
  
  void characterizeOpticalSystem(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, std::vector<OpticalSystem>& OS);
  void computeQ(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& Q);
  //void computeP(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase,
    //                  const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& P);
  void computeP(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase,
                      const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& P);  
  double objectiveFunction(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  void objectEstimate(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                              const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, const cv::Mat& P, 
                              const cv::Mat& Q, cv::Mat& F);
  void noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                   const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter);
  cv::Mat gradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  
private:
  cv::Mat F_;   //Object estimate
};

#endif /* METRIC_H_ */