/*
 * Minimization.h
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#ifndef MINIMIZATION_H_
#define MINIMIZATION_H_

#include <iostream>
#include <vector>
#include <limits>  //numeric_limit<double>::epsilon
#include <algorithm>    // std::max
#include <cmath>   //std::abs
#include <functional>   //function objects
#include "opencv2/opencv.hpp"
#include "CustomException.h"


//The process of finding the set of alpha values that gives the optimum phase aberration is non-linear
//here we aim to achieve a way to find the optimum increment in alpha values to be nearer to optimum solution in the next iteration
//The system described above IS linear and the new variable (x) is actually delta of alpha, which gives direction and magnitude towards the minimum
//and can be written as Ax-b=0, being dim(A)={MxM} and dim(b)={1xM}, and M the number of parameter we chose to define the aberration

//The result are two matrix, A and b that summarizes the whole linear equation system
//A matrix could be seen as Hessian matrix and b as the gradient vector
//enum class MinimizationMethod {Newton, BFGS , ConjugateGradients};
//

class Minimization
{
public:
  Minimization();
  virtual ~Minimization();
  
  double fret() const {return fret_;};
  
  void bracket(const double& a, const double& b, std::function<double(double)> &func);
  
  double brent(std::function<double(double)> &func);
  
  double linmin(cv::Mat& p, cv::Mat& xi, std::function<double(cv::Mat)> &func);
  
  //Build gradient and set next point a direction in convergence to the minimum
  void dfpmin(cv::Mat &p, int &iter, double &fret, 
              std::function<double(cv::Mat )> &func, std::function<cv::Mat(cv::Mat)> &dfunc);
  
  int nextStep(cv::Mat &p, cv::Mat &xi, cv::Mat &g, 
               cv::Mat &hessin, double &fret, std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc);
 
  void minimize(cv::Mat &p, const cv::Mat &Q2,
                const std::function<double(cv::Mat)>& func, const std::function<cv::Mat(cv::Mat)>& dfunc);
  
  
private:
  int iter_;   //total number of iterations to get to the mininum
  double fret_;  //function value at minimum
  
  double ax, bx, cx, fa, fb, fc;   //Variable used by bracket method
  
  double xmin,fmin;   //Variable used by brent 1D minimization method
	const double tol = 3.0e-8;   //Precision at which the minimum is found
  const double gtol = 3.0e-8;   //Precision gradient at which the minimum is found
  
  const int ITMAX = 200;  //maximum number of steps to reach the minimum

};




#endif /* MINIMIZATION_H_ */
