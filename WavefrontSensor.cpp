/*
 * WavefrontSensor.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#include "WavefrontSensor.h"
//#include "CustomException.h"
#include "Zernikes.h"
//#include <cmath>
#include "PDTools.h"
#include "TelescopeSettings.h"
//#include "FITS.h"
#include "Metric.h"
#include "Minimization.h"
#include <fstream>
//ANEXO
//How to get with python null space matrix from constraints, Q2
//import scipy
//import scipy.linalg
//A = scipy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
//Q, R = scipy.linalg.qr(A)

constexpr double PI = 3.14159265359;  //2 * acos(0.0);

//Other names, phaseRecovery, ObjectReconstruction, ObjectRecovery

WavefrontSensor::WavefrontSensor()
{
  diversityFactor_ = {0.0, -2.21209};
}

WavefrontSensor::~WavefrontSensor()
{
  // TODO Auto-generated destructor stub
}


cv::Mat
WavefrontSensor::WavefrontSensing(const std::vector<cv::Mat>& d, const std::vector<double>& meanPowerNoise)
{
  cv::Size d_size = d.front().size();
  for(cv::Mat di : d)
  {
    if (d_size != di.size())
    {
      std::cout << "Input dataset images must be iqual size" << std::endl;
      //throw CustomException("Input dataset images must be iqual size");
    }
  }

  TelescopeSettings tsettings(d_size.width);
  unsigned int numberOfZernikes = 14;   //total number of zernikes to be considered

  std::vector<cv::Mat> D;
  for(cv::Mat di : d)
  {
    cv::Mat Di;
    cv::dft(di, Di, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    fftShift(Di);
    D.push_back(Di);
  }

  unsigned int pupilSideLength = optimumSideLength(d_size.width/2, tsettings.pupilRadiousPixels());
  std::cout << "pupilRadiousPixels: " << tsettings.pupilRadiousPixels() << std::endl;
  
  //double pupilRadiousP = tsettings.pupilRadiousPixels();
  cv::Mat pupilAmplitude = Zernikes::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());
  std::vector<cv::Mat> zBase = Zernikes::zernikeBase(numberOfZernikes, d_size.width, tsettings.pupilRadiousPixels());
  
  Metric mm;
  //Objective function and gradient of the objective function
  std::function<double(cv::Mat)> func = std::bind(&Metric::objectiveFunction, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
  std::function<cv::Mat(cv::Mat)> dfunc = std::bind(&Metric::gradient, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
  
  int M = zBase.size();
  int K = D.size();
  cv::Mat Q2;
  partlyKnownDifferencesInPhaseConstraints(M, K, Q2);
  
  //Used this: http://davidstutz.de/matrix-decompositions/matrix-decompositions/householder/demo
  //double q2_oneCoeff[] = {0, 0, 0, -0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  //Q2 = cv::Mat(K*M, 1, cv::DataType<double>::type, q2_oneCoeff);
  
  //p: initial point; Q2: null space of constraints; objFunction: function to be minimized; gradFunction: first derivative of objFunction 
  cv::Mat p = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
  
  
  Minimization minimizationKit;
  minimizationKit.minimize(p, Q2, func, dfunc);
  std::cout << "mimumum: " << p.t() << std::endl;

  return mm.F();
}

void WavefrontSensor::ista(cv::Mat& u,  const std::function<double(cv::Mat)>& func, const std::function<cv::Mat(cv::Mat)>& grad)
{
  double t(1.0);
  double lambda(0.0001);
  auto shrink = [] (const double& x, const double& alpha) -> double
  {
    //return std::copysign(1.0, y) * std::max( std::abs(y)-alpha, 0.0);
    return std::copysign(1.0, x) * std::abs(x)-alpha;
  };
  cv::Mat y;
  u.copyTo(y);
  for(unsigned int hh = 0; hh<1000; ++hh)
  {
    std::cout << "func(u): " << func(u) << std::endl; 
    cv::Mat new_u = y - (splitComplex(grad(y)).first).mul(t);
    cv::Mat diff = new_u - u;
    new_u.copyTo(u);   //gradient at point u
    for(unsigned int i = 0; i<u.total(); ++i)
    {
      u.at<double>(0,i) = shrink(u.at<double>(0,i), lambda*t);
    }
    
    t = ( 1.0 + std::sqrt(1.0+(4*t*t)) )/2.0;
    y = u + ((t-1.0)/t)*(diff);
  }
  
  std::cout << "u: " << u << std::endl; 
}
