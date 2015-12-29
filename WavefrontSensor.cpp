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
#include "CompressedSensing.h"
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
  unsigned int numberOfZernikes = 28;   //total number of zernikes to be considered

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
  
  //Used this: http://davidstutz.de/matrix-decompositions/matrix-decompositions/householder
  //double q2_oneCoeff[] = {0, 0, 0, -0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  //Q2 = cv::Mat(K*M, 1, cv::DataType<double>::type, q2_oneCoeff);
  
  //p: initial point; Q2: null space of constraints; objFunction: function to be minimized; gradFunction: first derivative of objFunction 
  cv::Mat p = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
  
  
  Minimization minimizationKit;
  
  minimizationKit.minimize(p, Q2, func, dfunc, Minimization::Method::FISTA);
  std::cout << "mimumum: " << p.t() << std::endl;
  return mm.F();
}


/*
  //First attempt to solve through IHT
  auto vec = [](const std::vector<cv::Mat> &matrixV, cv::Mat& vector)
  {
    std::vector<cv::Mat> vv;
    for(size_t i = 0; i < matrixV.size(); ++i)
    {
      for(unsigned int j = 0; j < matrixV.at(i).cols; ++j)
      {
        vv.push_back(matrixV.at(i).col(j)); 
      }
    }
    cv::vconcat(vv, vector);
  };
  
  //Keep only k largest  coefficients
  auto hardThreshold = [](cv::Mat& p, const unsigned int& k)
  {
    cv::Mat mask = cv::Mat::ones(p.size(), CV_8U);
    cv::Mat pp(cv::abs(p));
    for(unsigned int i=0;i<k;++i)
    {
      cv::Point maxLoc;
      cv::minMaxLoc(pp, nullptr,nullptr, nullptr, &maxLoc, mask);
      mask.at<char>(maxLoc.y, maxLoc.x) = 0;
    }
    p.setTo(0.0, mask);
  };
  
  cv::Mat x0 = cv::Mat::zeros(K*M, 1, cv::DataType<double>::type);
  cv::Mat x0_imag = cv::Mat::zeros(K*M, 1, cv::DataType<double>::type);
  double mu(1.0);
  unsigned int sparsity(4);
  for(unsigned int j =0; j<500; ++j)
  {
    std::vector<cv::Mat> De;
    cv::Mat jacobian, measurement, observation;
    mm.phi(x0, D, zBase, meanPowerNoise, De);
    mm.compute_dphi(x0, D, zBase, meanPowerNoise, jacobian);
    vec(De, measurement);
    vec(D, observation);
    cv::Mat est = mu * (observation - measurement);
    cv::Mat zer = cv::Mat::zeros(est.size(), est.type());
    std::vector<cv::Mat> x0_ = {x0, x0_imag};
    cv::Mat x0_c;
    cv::merge(x0_, x0_c);
    cv::gemm(conjComplex(jacobian), est, -1.0, x0_c, 1.0, x0_c, cv::GEMM_1_T);
    x0 = splitComplex(x0_c).first;
    hardThreshold(x0, sparsity);
    
    std::cout << "x0: " << x0.t() << std::endl;
  }
*/