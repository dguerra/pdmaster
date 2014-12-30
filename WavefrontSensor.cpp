/*
 * WavefrontSensor.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#include "WavefrontSensor.h"
#include "CustomException.h"
#include "Zernikes.h"
#include "Zernikes.cpp"
#include <cmath>
//#include "NoiseEstimator.h"
//#include "WaveletTransform.h"
#include "PDTools.h"
//#include "Optimization.h"
//#include "OpticalSystem.h"
//#include "ErrorMetric.h"
#include "GetStep.h"
#include "TelescopeSettings.h"
#include "Metric.h"
#include "Minimization.h"

constexpr double PI = 2*acos(0.0);

//Other names, phaseRecovery, ObjectReconstruction, ObjectRecovery

WavefrontSensor::WavefrontSensor()
{
  maximumIterations_ = 50;
  diversityFactor_ = {0.0, -2.21209};
  lmMinimum_ = 100;
  iterationMinimum_ = 1;
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
      //std::cout << "Input dataset images must be iqual size" << std::endl;
      throw CustomException("Input dataset images must be iqual size");
    }
  }

  //unsigned long imgSize = d_size.width;
  TelescopeSettings tsettings(d_size.width);
  //MonitorLog monitorLog;
  unsigned int numberOfNonSingularities(0);
  double singularityThresholdOverMaximum(0.0);

  //c == recipients of zernike coefficients
  cv::Mat c = cv::Mat::zeros(14, 1, cv::DataType<double>::type);
  cv::Mat dc = cv::Mat::ones(c.size(), cv::DataType<double>::type);  //random initialization with rms greater than dcRMS_minimum
  cv::Mat zernikesInUse = cv::Mat::ones(c.size(), cv::DataType<bool>::type); //true where zernike index is used and false otherwise
  //zernikesInUse.at<bool>(0,0) = false;
  //zernikesInUse.at<bool>(0,1) = false;
  //zernikesInUse.at<bool>(0,2) = false;

  cv::Mat alignmentSetup = cv::Mat::zeros(c.size(), cv::DataType<bool>::type); //true for piston and tip-tilt coeffcients only
  //alignmentSetup.at<bool>(0,0) = true;
  //alignmentSetup.at<bool>(0,1) = true;
  //alignmentSetup.at<bool>(0,2) = true;

  unsigned int pupilSideLength = optimumSideLength(d_size.width/2, tsettings.pupilRadiousPixels());
  std::cout << "pupilSideLength: " << pupilSideLength << std::endl;
  std::map<unsigned int, cv::Mat> zernikeCatalog = Zernikes<double>::buildCatalog(c.total(), pupilSideLength, tsettings.pupilRadiousPixels());

  std::cout << "Total original image energy: " << cv::sum(d.front()) << std::endl;

  std::vector<cv::Mat> D;
  for(cv::Mat di : d)
  {
    cv::Mat Di;
    cv::dft(di, Di, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    shift(Di, Di, Di.cols/2, Di.rows/2);
    D.push_back(Di);
  }

  cv::Mat z4 = Zernikes<double>::phaseMapZernike(4, pupilSideLength, tsettings.pupilRadiousPixels());
  double z4AtOrigen = Zernikes<double>::pointZernike(4, 0, 0);
  std::vector<cv::Mat> diversityPhase;
  for(double dfactor : diversityFactor_)
  {
    //defocus zernike coefficient: c4 = dfactor * PI/(2.0*std::sqrt(3.0))
	  diversityPhase.push_back( (dfactor * PI/(2.0*std::sqrt(3.0))) * (z4 - z4AtOrigen));
  }
  double pupilRadiousP = tsettings.pupilRadiousPixels();
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, pupilRadiousP);
    
  static bool done(false);
  double lmPrevious(0.0);
  //iterate through this loop until stable solution of zernike coefficients is found
  for (unsigned int iteration = 0; iteration < maximumIterations_; ++iteration)
  {
      //little test to compare both old and new algorithms
  if(!done)
  {  //Check that every objectivefunction value fits the old algorithm
    done = true;
    Metric mm, mm_p;
    std::vector<double> meanPowerNoise = {2.08519e-09, 1.9587e-09};    //sample case
    std::vector<cv::Mat> zBase = Zernikes<double>::zernikeBase(c.total(), pupilSideLength, pupilRadiousP);
    
    Minimization minimizationKit;
    
    std::function<double(cv::Mat)> objFunction = std::bind(&Metric::objectiveFunction, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
    
    std::function<cv::Mat(cv::Mat)> gradFunction = std::bind(&Metric::gradient, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
    
    std::function<cv::Mat(cv::Mat)> dfuncc_diff = [objFunction] (cv::Mat x) -> cv::Mat
    { //make up gradient vector through slopes and tiny differences
      double EPS(1.0e-8);
      cv::Mat df = cv::Mat::zeros(x.size(), x.type());
  	  cv::Mat xh = x.clone();
  	  double fold = objFunction(x);
      for(unsigned int j = 0; j < x.total(); ++j)
      {
        double temp = x.at<double>(j,0);
        double h = EPS * std::abs(temp);
        if (h == 0) h = EPS;
        xh.at<double>(j,0) = temp + h;
        h = xh.at<double>(j,0) - temp;
        double fh = objFunction(xh);
        xh.at<double>(j,0) = temp;
        df.at<double>(j,0) = (fh-fold)/h;    
      }
      return df;  
    };
  
    int M = zBase.size();
    int K = D.size();
    cv::Mat Q2 = cv::Mat::eye(M*K, M*K, cv::DataType<double>::type);
    
    //cv::vconcat(cv::Mat::eye(M, M, cv::DataType<double>::type), cv::Mat::eye(M, M, cv::DataType<double>::type), Q2);
    //cv::multiply(Q2, 1.0/std::sqrt(2.0), Q2);
    
    //p: initial point; Q2: null space of constraints; objFunction: function to be minimized; gradFunction: first derivative of objFunction
    cv::Mat p;
    cv::vconcat(c.clone(), c.clone(), p);
    p = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
    std::cout << "objFunction(p) = " << objFunction(p) << std::endl;
    cv::Mat gF = gradFunction(p);
    std::cout << "gradFunction(p) = " << gF.t() << std::endl;
    cv::Mat dgF = dfuncc_diff(p);
    std::cout << "dfuncc_diff(p) = " << dgF.t() << std::endl;
    cv::Mat coc;
    cv::divide(gF, dgF, coc);
    std::cout << "coc: " << coc.t() << std::endl;
    minimizationKit.minimize(p, Q2, objFunction, gradFunction);
    std::cout << "mimumum: " << p.t() << std::endl;
  }

	  std::cout << "iteration: " << iteration << std::endl;
    int ret = getstep( c, D, diversityPhase, pupilAmplitude, pupilSideLength, zernikesInUse, alignmentSetup, zernikeCatalog,
    		               pupilRadiousP, meanPowerNoise, lmPrevious,
                       numberOfNonSingularities, singularityThresholdOverMaximum, dc);

    //very importan!! why do here have to substract instead of add?? what did I do wrong before?
    if(!ret)
    {
      c = c - dc;
    }
    else
    {
      break;
    }
  }

  return c;
}
/*
void WavefrontSensor::showRestore(ErrorMetric errMet, cv::Mat& fm)
{
  cv::Mat FMH;
  cv::mulSpectrums(errMet.FM(),errMet.noiseFilter(),FMH, cv::DFT_COMPLEX_OUTPUT);
  shift(FMH, FMH, FMH.cols/2, FMH.cols/2);
  cv::idft(FMH, fm, cv::DFT_REAL_OUTPUT);
//  cv::normalize(fm, fm, 0, 1, CV_MINMAX);
  //static int i(0);
  //cv::imshow("restored" + std::to_string(++i), fm);
//  cv::waitKey();
}
*/
