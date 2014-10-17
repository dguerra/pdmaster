/*
 * Linearization.cpp
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#include "Linearization.h"
#include "PDTools.h"

Linearization::Linearization()
{
  // TODO Auto-generated constructor stub

}

Linearization::~Linearization()
{
  // TODO Auto-generated destructor stub
}

void Linearization::computeHessianMatrix_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D)
{
   cv::Mat U;
   unsigned int i = 0;
   //cv::mulSpectrums(divComplex(D.at(i), Q)
}

void Linearization::computeGrandientVector_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D, const cv::Mat& F, const std::vector<cv::Mat>& zernikeBase)
{
  cv::Mat conjF  = conjComplex(F);
  cv::Mat absF   = absComplex(F);
  cv::Mat absF2  = absF.mul(absF);
  cv::Mat acc    = cv::Mat::zeros(absF.size(), absF.type());

  for(unsigned int k = 0; k < OS.size(); ++k)
  {
    cv::Mat pj, FDj, F2Sj, re, pjre, pjre_f, term;

    cv::idft(OS.at(k).generalizedPupilFunction(), pj, cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(conjF, D.at(k), FDj, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(absF2, OS.at(k).otf(), F2Sj, cv::DFT_COMPLEX_OUTPUT);
    cv::idft(FDj-F2Sj, re, cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(pj, re, pjre, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(pjre, pjre_f, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    cv::mulSpectrums(conjComplex(OS.at(k).generalizedPupilFunction()), pjre_f, term, cv::DFT_COMPLEX_OUTPUT);
    acc += splitComplex(term).second;   //only takes imaginary part of the term

  }

  acc = -2.0 * acc;

  cv::Mat b(zernikeBase.size(), 1, cv::DataType<double>::type);
  for(unsigned int i = 0; i < zernikeBase.size(); ++i)
  {
    b.at<double>(i, 0) = zernikeBase.at(i).dot(acc);   //inner product between acc variable and every element of the zernike base
  }
}
