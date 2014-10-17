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
#include "NoiseEstimator.h"
#include "WaveletTransform.h"
#include "PDTools.h"
#include "Optimization.h"
#include "OpticalSystem.h"
#include "ErrorMetric.h"
#include "TelescopeSettings.h"
#include "MonitorLog.h"

constexpr double PI = 2*acos(0.0);

//Other names, phaseRecovery, ObjectReconstruction, ObjectRecovery

WavefrontSensor::WavefrontSensor()
{
  dcRMS_Minimum_ = 1.0e-2;
  lmIncrement_Minimum_ = 0.0;
  maximumIterations_ = 50;
  imageCoreSize_ = 70;
  diversityFactor_ = -2.21209;
  lmMinimum_ = 100;
  iterationMinimum_ = 1;
}

WavefrontSensor::~WavefrontSensor()
{
  // TODO Auto-generated destructor stub
}


cv::Mat
WavefrontSensor::WavefrontSensing(const cv::Mat& d0, const cv::Mat& dk, const double& meanPowerNoiseD0, const double& meanPowerNoiseDk)
{
  if (d0.cols != dk.cols || d0.rows != dk.rows || d0.rows != d0.cols)
  {
    throw CustomException("Images focused and defocused must be iqual size");
  }

  double filterTuning_ = 1.0;
  unsigned long imgSize = d0.cols;
  TelescopeSettings tsettings(imgSize);
  MonitorLog monitorLog;
  unsigned int numberOfNonSingularities(0);
  double singularityThresholdOverMaximum(0.0);

  //c == recipients of zernike coefficients
  cv::Mat c = cv::Mat::zeros(14, 1, cv::DataType<double>::type);
  cv::Mat dc = cv::Mat::ones(c.size(), cv::DataType<double>::type);  //random initialization with rms greater than dcRMS_minimum
  cv::Mat zernikesInUse = cv::Mat::ones(c.size(), cv::DataType<bool>::type); //true where zernike index is used and false otherwise
  zernikesInUse.at<bool>(0,0) = false;
  zernikesInUse.at<bool>(0,1) = false;
  zernikesInUse.at<bool>(0,2) = false;

  cv::Mat alignmentSetup = cv::Mat::zeros(c.size(), cv::DataType<bool>::type); //true for piston and tip-tilt coeffcients only
  alignmentSetup.at<bool>(0,0) = true;
  alignmentSetup.at<bool>(0,1) = true;
  alignmentSetup.at<bool>(0,2) = true;

  unsigned int pupilSideLength = optimumSideLength(imgSize/2, tsettings.pupilRadiousPixels());
  std::cout << "pupilSideLength: " << pupilSideLength << std::endl;
  std::map<unsigned int, cv::Mat> zernikeCatalog = Zernikes<double>::buildCatalog(c.total(), pupilSideLength, tsettings.pupilRadiousPixels());

  std::cout << "Total original image energy: " << cv::sum(d0) << std::endl;

  cv::Mat D0, Dk;
  cv::dft(d0, D0, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  shift(D0, D0, D0.cols/2, D0.rows/2);
  cv::dft(dk, Dk, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  shift(Dk, Dk, Dk.cols/2, Dk.rows/2);

  double c4 = diversityFactor_ * PI/(2.0*std::sqrt(3.0));
  std::cout << "c4: " << c4 << std::endl;
  cv::Mat z4 = Zernikes<double>::phaseMapZernike(4, pupilSideLength, tsettings.pupilRadiousPixels());
  double z4AtOrigen = Zernikes<double>::pointZernike(4, 0, 0);
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());
  cv::Mat focusedPupilPhase, defocusedPupilPhase;
  double lmPrevious(0.0);
  //iterate through this loop until stable solution of zernike coefficients is found
  for (unsigned int iteration = 0; iteration < maximumIterations_; ++iteration)
  {
    focusedPupilPhase = Zernikes<double>::phaseMapZernikeSum(pupilSideLength, tsettings.pupilRadiousPixels(), c.setTo(0, 1-zernikesInUse ));
    defocusedPupilPhase = focusedPupilPhase + c4*(z4-z4AtOrigen);
    // + c2  * Zernikes<double>::phaseMapZernike(2, imgSize/2, tsettings.pupilRadiousPixels())   //tiptilt X
    // + c3  * Zernikes<double>::phaseMapZernike(3, imgSize/2, tsettings.pupilRadiousPixels())   //tiptilt Y

    OpticalSystem focusedOS(focusedPupilPhase, pupilAmplitude);
    OpticalSystem defocusedOS(defocusedPupilPhase, pupilAmplitude);

    ErrorMetric EM(focusedOS, defocusedOS, D0, Dk, meanPowerNoiseD0*filterTuning_, meanPowerNoiseDk*filterTuning_, zernikeCatalog, zernikesInUse);
    cv::Mat fm;
    //showRestore(EM, fm);
    std::cout << "Total restored image energy: " << cv::sum(fm) << std::endl;
    cv::Mat eCoreZeroMean = backToImageSpace(EM.E(), cv::Size(imageCoreSize_, imageCoreSize_));
    std::vector<cv::Mat> dedcCoreZeroMean;
    for(cv::Mat dEdci : EM.dEdc()) dedcCoreZeroMean.push_back(backToImageSpace(dEdci, cv::Size(imageCoreSize_, imageCoreSize_)));

    double lmCurrent = cv::sum(eCoreZeroMean.mul(eCoreZeroMean)).val[0]/eCoreZeroMean.total();
    double lmIncrement = std::abs(lmCurrent - lmPrevious)/lmCurrent;

    cv::Mat c2, dc2;
    cv::pow(c.setTo(0, alignmentSetup), 2.0, c2);
    cv::pow(dc.setTo(0, alignmentSetup), 2.0, dc2);
    double cRMS = std::sqrt(cv::sum(c2).val[0]);
    double dcRMS = std::sqrt(cv::sum(dc2).val[0]);

    Record rec(iteration, lmCurrent, lmIncrement, c, cRMS, dcRMS, numberOfNonSingularities, singularityThresholdOverMaximum);
    rec.printValues();
    monitorLog.add(rec);

    lmPrevious = lmCurrent;

    if((dcRMS < dcRMS_Minimum_) || (lmIncrement < lmIncrement_Minimum_))
    {
      if(dcRMS < dcRMS_Minimum_) std::cout << "dcRMS_Minimum has been reached." << std::endl;
      if(lmIncrement < lmIncrement_Minimum_) std::cout << "lmIncrement_Minimum has been reached." << std::endl;
      break;
    }
    //Also add the case where tip-tilt rms are lower than minimum
    //  (std::sqrt(dc(2)*dc(2)+dc(3)*dc(3)) < dcRMS_Minimum_) ||

    //e, and dedc must be taken from central subfield and mean zero here before optimization
    Optimization optimumIncrement(eCoreZeroMean, dedcCoreZeroMean);

    numberOfNonSingularities = optimumIncrement.numberOfNonSingularities();
    singularityThresholdOverMaximum = optimumIncrement.singularityThresholdOverMaximum();

    dc = optimumIncrement.dC();
    //very importan!! why do here have to substract instead of add?? what did I do wrong before?
    c = c - dc;

  }

  return focusedPupilPhase;
}

cv::Mat WavefrontSensor::backToImageSpace(const cv::Mat& fourierSpaceMatrix, const cv::Size& centralROI)
{
  cv::Mat imageMatrixROIZeroMean;
  if(!fourierSpaceMatrix.empty())
  {
    cv::Mat imageMatrix;
    cv::Mat fourierSpaceMatrixShift(fourierSpaceMatrix);
    //shift quadrants back to origin in the corner, inverse transform, take central region, force zero-mean
    shift(fourierSpaceMatrixShift, fourierSpaceMatrixShift, fourierSpaceMatrixShift.cols/2, fourierSpaceMatrixShift.rows/2);
    cv::idft(fourierSpaceMatrixShift, imageMatrix, cv::DFT_REAL_OUTPUT);
    cv::Mat imageMatrixROI = takeoutImageCore(imageMatrix, centralROI.height);
    imageMatrixROIZeroMean = imageMatrixROI - cv::mean(imageMatrixROI);
  }
  return imageMatrixROIZeroMean;
}

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
