/*
 * Optimization.h
 *
 *  Created on: Feb 13, 2014
 *      Author: dailos
 */

#ifndef OPTIMIZATION_H_
#define OPTIMIZATION_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

class Optimization
{
public:
  Optimization();
  Optimization(const cv::Mat& E, const std::vector<cv::Mat>& dEdc);
  virtual ~Optimization();
  cv::Mat dC()const {return dC_;};
  int numberOfNonSingularities()const{return numberOfNonSingularities_;};
  double singularityThreshold()const{return singularityThreshold_;};
  double singularityThresholdOverMaximum()const{return singularityThresholdOverMaximum_;};
  double cRMS()const{return cRMS_;};
private:
  double singularityThresholdOverMaximum_;  //lower limit to consider singularity in SVD method
  double cRMSMinimum_;  //lower limit of rms value of result while applying SVD method
  int nonZeroSingularitiesMinimum_;
  int numberOfNonSingularities_;    //number of singularities in the process
  double singularityThreshold_;   //singularityThreshold
  double cRMS_;
  cv::Mat dC_;

  void compute_matrixA_(const std::vector<cv::Mat>& dEdc, cv::Mat& A);
  void compute_matrixB_(const std::vector<cv::Mat>& dEdc, const cv::Mat& E, cv::Mat& B);
};
#endif /* OPTIMIZATION_H_ */

