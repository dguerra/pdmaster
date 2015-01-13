/*
 * Optimization.cpp
 *
 *  Created on: Feb 13, 2014
 *      Author: dailos
 */

#include "Optimization.h"
#include "MatrixEquation.h"
#include "PDTools.h"

//Matrix equation solver
//This module solves matrix equations such as: Ax-b=0; where dim(A)={MxM} and dim(b)={1xM}

Optimization::Optimization()
{
  nonZeroSingularitiesMinimum_ = 6;
  singularityThresholdOverMaximum_ = 0.04;
  cRMSMinimum_ = 0.8;
  cRMS_ = 1.0;
  singularityThreshold_ = 0.0;
  numberOfNonSingularities_ = 0;
}

Optimization::Optimization(const cv::Mat& E, const std::vector<cv::Mat>& dEdc)
{
  nonZeroSingularitiesMinimum_ = 6;
  singularityThresholdOverMaximum_ = 0.04;
  cRMSMinimum_ = 0.8;

  //Elements are now in image domain!!
  cv::Mat A_(dEdc.size(), dEdc.size(), cv::DataType<double>::type), B_(dEdc.size(), 1, cv::DataType<double>::type);

  compute_matrixA_(dEdc, A_);
  compute_matrixB_(dEdc, E, B_);
  //solve_system(A_, B_, dC_);

  //Test equivalence
  MatrixEquation mEq;
  mEq.solve(A_, B_, dC_);    //Solve A_dC+B_=0, where dC is the unknown
  singularityThresholdOverMaximum_ = mEq.singularityThresholdOverMaximum();
  cRMS_ = mEq.xRMS();
  singularityThreshold_ = singularityThreshold();
  numberOfNonSingularities_ = mEq.numberOfNonSingularities();
}

Optimization::~Optimization()
{
  // TODO Auto-generated destructor stub
}

void Optimization::compute_matrixA_(const std::vector<cv::Mat>& dEdc, cv::Mat& A)
{
  //it has to be apodized and measure space images!!

  for(auto dEdci = dEdc.cbegin(), dEdci_end = dEdc.cend(); dEdci != dEdci_end; ++dEdci)
  {
    for(auto dEdcj = dEdc.cbegin(), dEdcj_end = dEdc.cend(); dEdcj != dEdcj_end; ++dEdcj)
    {
      if(!(*dEdci).empty() && !(*dEdcj).empty())
      {
        A.at<double>(std::distance(dEdc.cbegin(), dEdci),std::distance(dEdc.cbegin(), dEdcj)) = dEdcj->dot(*dEdci);
      }
      else
      {
        A.at<double>(std::distance(dEdc.cbegin(), dEdci),std::distance(dEdc.cbegin(), dEdcj)) = 0.0;
      }
    }

  }

  A = A/(dEdc.back().total());
}


void Optimization::compute_matrixB_(const std::vector<cv::Mat>& dEdc, const cv::Mat& E, cv::Mat& B)
{
  for(auto dEdci = dEdc.cbegin(), dEdci_end = dEdc.cend(); dEdci != dEdci_end; ++dEdci)
  {
    if(!(*dEdci).empty())
    {
      B.at<double>(std::distance(dEdc.cbegin(), dEdci), 0) = E.dot(*dEdci);
    }
    else
    {
      B.at<double>(std::distance(dEdc.cbegin(), dEdci), 0) = 0.0;
    }
  }
  B = -B/(E.total());
}
