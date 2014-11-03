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

template <class T>
void lnsrch(cv::Mat &xold, const double fold, cv::Mat &g, cv::Mat &p,
    cv::Mat &x, double &f, const double stpmax, bool &check, T &func);

void Linearization::computeGrandientVector_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase)
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


  cv::Mat conjF  = conjComplex(F);
  cv::Mat absF   = absComplex(F);
  cv::Mat absF2  = absF.mul(absF);

  /////////////////////////////////////////////////
  //Compute gradient vector, b, with N = K*M elements
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


template <class T>
void lnsrch(cv::Mat &xold, const double fold, cv::Mat &g, cv::Mat &p,
    cv::Mat &x, double &f, const double stpmax, bool &check, T &func)
{
  const double ALF = 1.0e-4, TOLX = std::numeric_limits<double>::epsilon();
  double a, alam, alam2 = 0.0, alamin, b, disc, f2 = 0.0;
  double rhs1, rhs2, slope = 0.0, sum = 0.0, temp, test, tmplam;
  unsigned int i, n = xold.total();  //It's a vector, so total number of elements is the size
  check = false;
  sum = p.dot(p);
  sum = std::sqrt(sum);

  if (sum > stpmax)
  {
    p = p.mul(stpmax/sum);
  }

  slope = g.dot(p);
  if (slope >= 0.0)
  {
    throw("Roundoff problem in lnsrch.");
  }

  test = 0.0;
  for (i=0; i<n; i++)
  {
    temp = std::abs(p.at<double>(i,0))/std::max(std::abs(xold.at<double>(i,0)),1.0);
    if (temp > test) test = temp;
  }
  alamin = TOLX / test;
  alam = 1.0;
  for (;;)
  {
    x = xold + (alam * p);

    f = func(x);
    if (alam < alamin)
    {
      xold.copyTo(x);
      check = true;
      return;
    }
    else if (f <= fold+ALF*alam*slope) return;
    else
    {
      if (alam == 1.0)
      {
        tmplam = -slope/(2.0*(f-fold-slope));
      }
      else
      {
        rhs1 = f-fold-alam*slope;
        rhs2 = f2-fold-alam2*slope;
        a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
        b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
        if (a == 0.0) tmplam = -slope/(2.0*b);
        else
        {
          disc = b * b - 3.0 * a * slope;
          if (disc < 0.0)
          {
            tmplam = 0.5 * alam;
          }
          else if (b <= 0.0) tmplam = (-b + std::sqrt(disc))/(3.0 * a);
          else tmplam = -slope/(b + std::sqrt(disc));
        }
        if (tmplam > 0.5 * alam)
        {
          tmplam = 0.5 * alam;
        }
      }
    }
    alam2 = alam;
    f2 = f;
    alam = std::max(tmplam, 0.1 * alam);
  }
}

