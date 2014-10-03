/*
 * SubimageLayout.cpp
 *
 *  Created on: Jan 27, 2014
 *      Author: dailos
 */

#include "SubimageLayout.h"
#include "WavefrontSensor.h"
#include "NoiseEstimator.h"
#include "OpticalSystem.h"
#include "FITS.h"
#include "Zernikes.h"
#include "Zernikes.cpp"
#include "TelescopeSettings.h"
#include "WaveletTransform.h"

SubimageLayout::SubimageLayout()
{
  subimageSize_ = 128;  //size of the box to be analized
  subimageStepXSize_ = 0;
  subimageStepYSize_ = 0;
  apodizationPercent_ = 12.5;
}

SubimageLayout::~SubimageLayout()
{
  // TODO Auto-generated destructor stub
}

void SubimageLayout::navigateThrough()
{
  //nextSubimage()
  //The following actions should be done by the class in every sub-image
  cv::Mat dat, img;
  readFITS("/home/dailos/PDPairs/12-06-2013/pd.004.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);

  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  int X(100),Y(100);
  cv::Rect rect1(X, Y, subimageSize_, subimageSize_);
  cv::Rect rect2(936+X, 0+Y, subimageSize_, subimageSize_);

  cv::Mat d0 = img(rect1).clone();
  cv::Mat dk = img(rect2).clone();

  cv::Mat d0_norm;
  cv::normalize(d0, d0_norm, 0, 1, CV_MINMAX);
  cv::imshow("d0", d0_norm);

  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d0);
  noiseDefocused.meanPowerSpectrum(dk);

  std::cout << "noiseFocused.meanPower(): " << noiseFocused.meanPower() << std::endl;
  std::cout << "noiseFocused.sigma(): " << noiseFocused.sigma() << std::endl;

  //create apodization window and substract constant to have zero mean in the apodized image
  cv::Mat apodizationWindow;
  createModifiedHanningWindow(apodizationWindow, d0.cols, apodizationPercent_, cv::DataType<double>::type);
  cv::Scalar imageD0Offset = cv::sum(d0.mul(apodizationWindow))/cv::sum(apodizationWindow);
  cv::Scalar imageDkOffset = cv::sum(dk.mul(apodizationWindow))/cv::sum(apodizationWindow);

  WavefrontSensor wSensor;
  cv::Mat phase = wSensor.WavefrontSensing((d0-imageD0Offset).mul(apodizationWindow), (dk-imageDkOffset).mul(apodizationWindow),
                                   noiseFocused.meanPower(), noiseDefocused.meanPower());

  TelescopeSettings tsettings(subimageSize_);
  cv::Mat amplitude = Zernikes<double>::phaseMapZernike(1, phase.cols, tsettings.pupilRadiousPixels());
  OpticalSystem foc(phase, amplitude);
}


void SubimageLayout::createModifiedHanningWindow(cv::Mat& modifiedHanningWindow, const int& sideLength, const double& apodizedAreaPercent, int datatype)
{
  int apodizedArea = int((apodizedAreaPercent * sideLength) / 100);
  std::cout << "apodizedArea: " << apodizedArea << std::endl;
  cv::Mat hann;
  cv::createHanningWindow(hann, cv::Size(apodizedArea * 2, 3), datatype);
  cv::Mat modifiedHanningSlice = cv::Mat::ones(1,sideLength, datatype);
  (hann(cv::Rect(0,1,apodizedArea,1))).copyTo(modifiedHanningSlice(cv::Rect(0,0,apodizedArea,1)));
  (hann(cv::Rect(apodizedArea,1,apodizedArea,1))).copyTo(modifiedHanningSlice(cv::Rect((modifiedHanningSlice.cols-apodizedArea),0,apodizedArea,1)));
  //Matrix multiplications of one single colum 1xN by one single row Nx1 matrices, to create a NxN
  modifiedHanningWindow = modifiedHanningSlice.t() * modifiedHanningSlice;
}
