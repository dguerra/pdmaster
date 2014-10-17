/*
 * MatrixEquation.cpp
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#include "MatrixEquation.h"
#include "PDTools.h"

//Matrix equation solver
//This module solves matrix equations such as: Ax+b=0; where dim(A)={MxM} and dim(b)={1xM}

MatrixEquation::MatrixEquation()
{
  //SVD method parameters
  nonZeroSingularitiesMinimum_ = 6;
  singularityThresholdOverMaximum_ = 0.04;
  xRMSMinimum_ = 0.8;
  xRMS_ = 1.0;
  singularityThreshold_ = 0.0;
  numberOfNonSingularities_ = 0;
}

MatrixEquation::~MatrixEquation()
{
  // TODO Auto-generated destructor stub
}

void MatrixEquation::solve(const cv::Mat& A, const cv::Mat& b, cv::Mat& x)
{
  //Solve the system by using singular value decomposition method
  cv::Mat w, u, vt, ws;
  cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);

  //we suppose w values are ordered so first one is the maximum
  double maxVal(0.0), minVal(0.0);
  cv::minMaxIdx(w, nullptr, &maxVal);
  singularityThreshold_ = maxVal * singularityThresholdOverMaximum_;
  w.setTo(0, w < singularityThreshold_);
  numberOfNonSingularities_ = cv::countNonZero(w);

  for(;;)
  {
    cv::SVD::backSubst(w, u, vt, b, x);  //back substitution after zero-out element below threshold

    xRMS_ = std::sqrt(cv::sum((x).mul(x)).val[0]);
    numberOfNonSingularities_ = cv::countNonZero(w);   //total number of zero elements in the diagonal

    if(xRMS_ <= xRMSMinimum_ || numberOfNonSingularities_ <= nonZeroSingularitiesMinimum_)
    {
      break;
    }
    else
    {
      w.row(numberOfNonSingularities_-1) = cv::Scalar(0.0);   //sets last non-zero value to zero before start over again
    }
  }

  if(xRMS_ > xRMSMinimum_)
  { // loop broke due to low number of no singularities
    x = x * (xRMSMinimum_/xRMS_);
    xRMS_ = xRMSMinimum_;
  }

  cv::minMaxIdx(w, &minVal, &maxVal, nullptr, nullptr, w!=0);
  singularityThresholdOverMaximum_ = minVal/maxVal;

}
