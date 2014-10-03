/*
 * MatrixEquation.h
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#ifndef MATRIXEQUATION_H_
#define MATRIXEQUATION_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

class MatrixEquation
{
public:
  MatrixEquation();
  virtual ~MatrixEquation();
  cv::Mat x()const {return x_;};
  int numberOfNonSingularities()const{return numberOfNonSingularities_;};
  double singularityThreshold()const{return singularityThreshold_;};
  double singularityThresholdOverMaximum()const{return singularityThresholdOverMaximum_;};
  double xRMS()const{return xRMS_;};
  void solve(const cv::Mat& A, const cv::Mat& b, cv::Mat& x);

private:
  double singularityThresholdOverMaximum_;  //lower limit to consider singularity in SVD method
  double xRMSMinimum_;  //lower limit of rms value of result while applying SVD method
  int nonZeroSingularitiesMinimum_;
  int numberOfNonSingularities_;    //number of singularities in the process
  double singularityThreshold_;   //singularityThreshold
  double xRMS_;
  cv::Mat x_;    //Solution to the equation
};



#endif /* MATRIXEQUATION_H_ */
