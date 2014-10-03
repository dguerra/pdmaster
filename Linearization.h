/*
 * Linearization.h
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#ifndef LINEARIZATION_H_
#define LINEARIZATION_H_

#include <iostream>
#include <vector>
#include "OpticalSystem.h"
#include "opencv2/opencv.hpp"

//The process of finding the set of alpha values that gives the optimum phase aberration is non-linear
//here we aim to achieve a way to find the optimum increment in alpha values to be nearer to optimum solution in the next iteration
//The system described above IS linear and the new variable (x) is actually delta of alpha, which gives direction and magnitude towards the minimum
//and can be written as Ax-b=0, being dim(A)={MxM} and dim(b)={1xM}, and M the number of parameter we chose to define the aberration

//The result are two matrix, A and b that summarizes the whole linear equation system
//A matrix could be seen as Hessian matrix and b as the gradient vector
class Linearization
{
public:
  Linearization();
  enum class LinearizationMethod {Newton, LevenbergMarquardt, ConjugateGradients};
  virtual ~Linearization();
  //represents A matrix in equation: Ax-b=0
  void computeHessianMatrix_(const std::vector<OpticalSystem>& frameOS, const std::vector<cv::Mat>& dataFrame);

  //represents b vector in equation: Ax-b=0
  void computeGrandientVector_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D, const cv::Mat& F, const std::vector<cv::Mat>& zernikeCatalog);
private:
  cv::Mat A_;   //A is a MxM matrix
  cv::Mat b_;   //b is a 1xM matrix
};

#endif /* LINEARIZATION_H_ */
