/*
 * NoiseFilter.h
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */

#ifndef METRIC_H_
#define METRIC_H_

#include <iostream>
#include "Optics.h"
#include "opencv2/opencv.hpp"
#include <memory>
#include "Zernike.h"
class Metric
{
public:
  Metric(const std::vector<cv::Mat>& D, const double& pupilRadiousPxls, const unsigned int& numberOfElements, const double& meanPowerNoise);
  Metric(const std::vector<cv::Mat>& D, const std::shared_ptr<Zernike>& zrnk, const double& meanPowerNoise);
  virtual ~Metric();
  //Getters
  cv::Mat F() const {return F_;};
  
  //ComputeSi
  void characterizeOpticalSystem(const cv::Mat& coeffs, std::vector<Optics>& OS);
  void computeQ(const cv::Mat& coeffs, const std::vector<Optics>& OS, cv::Mat& Q);
  void computeP(const cv::Mat& coeffs, const std::vector<Optics>& OS, cv::Mat& P);
  
  void compute_dQ(const cv::Mat& zernikeElement, const std::vector<Optics>& OS, const unsigned int& j, cv::Mat& dQ);
  void compute_dP(const cv::Mat& zernikeElement, const std::vector<Optics>& OS, const unsigned int& j, cv::Mat& dP);
  
  void compute_dSj(const Optics& osj, const cv::Mat& zernikeElement, cv::Mat& dSj);

  //Objective funtion is |D_0 - F * S_0|^2 + |D_1 - F * S_1|^2
  double objective(const cv::Mat& coeffs);
  cv::Mat gradient(const cv::Mat& coeffs);
  //Compute Φ(x) and Jacobian of Φ in the equation y = Φ(x) + e
  void phi(      const cv::Mat& coeffs, std::vector<cv::Mat>& De );
  void phi(      const cv::Mat& coeffs, cv::Mat& De );
  void jacobian( const cv::Mat& coeffs, std::vector<std::vector<cv::Mat> >& jacob );
  void jacobian( const cv::Mat& coeffs, cv::Mat& jacob );
  void noiseFilter(const cv::Mat& coeffs, const double& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter);

private:
  cv::Mat F_;   //Object estimate
  std::shared_ptr<Zernike> zrnk_; 
  const double meanPowerNoise_;    //noise mean power spectrum, cannot be changed once initialized
  std::vector<cv::Mat> D_;   //data images
  //cv::Mat cutoff_mask_;    
};

#endif /* METRIC_H_ */