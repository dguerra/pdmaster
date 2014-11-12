//============================================================================
// Name        : PDPlayground.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <functional>   // std::minus
#include <numeric>      // std::accumulate
#include <math.h>
#include <tuple>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <algorithm>
#include "Zernikes.h"
#include "NoiseEstimator.h"
#include "WaveletTransform.h"
#include "PDTools.h"
#include "Optimization.h"
#include "OpticalSystem.h"
#include "ErrorMetric.h"
#include "opencv2/opencv.hpp"
#include "TestRoom.h"
#include "TelescopeSettings.h"
#include "WavefrontSensor.h"
#include "FITS.h"
#include "SubimageLayout.h"
#include "AWMLE.h"
#include "Minimization.h"

#include <chrono>

//to compare the results mean squared error (MSE) and the structural similarity index (SSIM)

struct Func
{
  double operator()(cv::Mat_<double> x)
  {
    //x^6 - 3*(x+1)^5 + 5 + (y+1)^6+y^5
    return std::pow(x.at<double>(0,0),6) - 3 * std::pow(x.at<double>(0,0)+1,5) + 5 + 
           std::pow(x.at<double>(1,0)+1,6)+ std::pow(x.at<double>(1,0),5);
    //return 1 + 4*(x.at<double>(0,0)+5)*x.at<double>(0,0) + 1*(x.at<double>(1,0)+3)*x.at<double>(1,0);
    //return x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(1,0)*x.at<double>(1,0);
  }
};

struct Dfunc
{
  cv::Mat_<double> operator()(cv::Mat_<double> x)
  {
    cv::Mat_<double> z(2,1);  //Size(2,1)->1 row, 2 colums
    z.at<double>(0,0) = 6 * std::pow(x.at<double>(0,0),5) - 15 * std::pow(x.at<double>(0,0)+1,4);
    z.at<double>(1,0) = 5 * std::pow(x.at<double>(1,0),4) + 6 * std::pow(x.at<double>(1,0)+1,5);
    //z.at<double>(0,0) = 8 * x.at<double>(0,0) + 20;
    //z.at<double>(1,0) = 2 * x.at<double>(1,0) + 3;
    //z.at<double>(0,0) = 2 * x.at<double>(0,0);
    //z.at<double>(1,0) = 2 * x.at<double>(1,0);
    return z.clone();
  }
};

int main()
{
  try
  {
    SubimageLayout subimageLayout;
    subimageLayout.navigateThrough();
/*
    Func f;
    Dfunc df;
    Minimization mm;
    cv::Mat_<double> p(2,1);
    p.at<double>(0,0) = 0;
    p.at<double>(1,0) = 0.8;
     
    int iter;
    double fret;
    mm.dfpmin(p, 3.0e-8, iter, fret, f, df);
    
    std::cout << "fret: " << fret << std::endl;
    std::cout << "p: " << p << std::endl;
*/    
  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}
