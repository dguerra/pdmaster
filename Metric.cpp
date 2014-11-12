/*
 * Metric.cpp
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */
#include "Metric.h"
#include "PDTools.h"

Metric::Metric()
{
  // TODO Auto-generated constructor stub

}

Metric::~Metric()
{
  // TODO Auto-generated destructor stub
}

//void Metric::computeObjectiveFunction

void Metric::computeGrandient_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase)
{
  double gamma(0.0);  //We are not going to consider this gamma value so we set to zero
  unsigned int K = OS.size();
  unsigned int M = zernikeBase.size();
  //Compute first Q value, needed to know the object estimate
  cv::Mat Q = cv::Mat::zeros(OS.front().otf().size(), OS.front().otf().type());

  for(OpticalSystem OSj : OS)
  {
    cv::Mat absSj = absComplex(OSj.otf());
    Q += absSj.mul(absSj);
  }
  Q = Q + gamma;

  //Compute now the object estimate, using Q
  cv::Mat F = cv::Mat::zeros(D.front().size(), D.front().type());
  cv::Mat acc = cv::Mat::zeros(D.front().size(), D.front().type());

  for(unsigned int k = 0; k < D.size(); ++k)
  {
    cv::Mat SjDj;
    cv::mulSpectrums(conjComplex(OS.at(k).otf()), D.at(k), SjDj, cv::DFT_COMPLEX_OUTPUT);
    acc += SjDj;
  }
  cv::Mat Q_1;
  cv::pow(Q, -1.0, Q_1);  //Q is a real matrix
  cv::mulSpectrums(makeComplex(Q_1), acc, F, cv::DFT_COMPLEX_OUTPUT);


  /////////////////////////////////////////////////
  //Compute gradient vector, b, with N = K*M elements
  cv::Mat conjF  = conjComplex(F);
  cv::Mat absF   = absComplex(F);
  cv::Mat absF2  = absF.mul(absF);

  cv::Mat b(cv::Size(K * M, 1), cv::DataType<double>::type);

  for(unsigned int k = 0; k < OS.size(); ++k)
  {
    cv::Mat P, pj, FDj, F2Sj, re, pjre, pjre_f, term;

    P = OS.at(k).generalizedPupilFunction();
    cv::idft(P, pj, cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(conjF, D.at(k), FDj, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(absF2, OS.at(k).otf(), F2Sj, cv::DFT_COMPLEX_OUTPUT);
    cv::idft(FDj-F2Sj, re, cv::DFT_REAL_OUTPUT);   //this term turn out to be real after the inverse fourier transform
    cv::multiply(pj, re, pjre);   //both pj and re are real matrices, so multiply in the other way

    cv::dft(pjre, pjre_f, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    cv::mulSpectrums(conjComplex(P), pjre_f, term, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat acc = (splitComplex(term).second).mul(-2 * std::sqrt(K));   //only takes imaginary part of the term

    for(unsigned int m = 0; m < M; ++m)
    {
      b.at<double>((k * M) + m, 0) = acc.dot(zernikeBase.at(m));
    }
  }

}
