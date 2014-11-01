#include <iostream>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"


template <class T>
void lnsrch(cv::Mat &xold, const double fold, cv::Mat &g, cv::Mat &p,
    cv::Mat &x, double &f, const double stpmax, bool &check, T &func);
  
int quasi_main()
{
  cv::Mat m = cv::Mat::ones(cv::Size(3,2), cv::DataType<double>::type);
  std::cout << "Hello: " << m.total() << std::endl;
  std::cout << "Hello: " << std::endl;
  return 0;
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
