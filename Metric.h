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
  void compute_dQ(const std::vector<cv::Mat>& D, const cv::Mat& zernikeElement, 
                        const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dQ);
  void computeP(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase,
                      const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& P);  
  void compute_dP(const std::vector<cv::Mat>& D, const cv::Mat& zernikeElement, 
                        const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dP);
  void phi( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                                  const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, std::vector<cv::Mat>& De );
  void compute_dphi( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                                  const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, std::vector<std::vector<cv::Mat> >& jacob );
  void compute_dphi( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                           const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, cv::Mat& jacob );                                  
  void compute_dSj(const OpticalSystem& osj, const cv::Mat& zernikeElement, cv::Mat& dSj);
  double objectiveFunction(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  void noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                   const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter);
  cv::Mat gradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise);
  
private:
  cv::Mat F_;   //Object estimate
};

#endif /* METRIC_H_ */