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
#include "TelescopeSettings.h"
#include "ImageQualityMetric.h"
#include "PDTools.h"

SubimageLayout::SubimageLayout()
{
  subimageSize_ = 42; //70; //56; //size of the box to be analized
  subimageStepXSize_ = 0;
  subimageStepYSize_ = 0;
  apodizationPercent_ = 12.5;
}

SubimageLayout::~SubimageLayout()
{
  // TODO Auto-generated destructor stub
}


void SubimageLayout::computerGeneratedImage()
{
  //Benchmark
  int isize = 42; //32; //128;
  cv::Mat img;
  if(true)
  {
    cv::Mat dat;
    readFITS("../inputs/surfi000.fits", dat);
    dat.convertTo(img, cv::DataType<double>::type);

    int X(100),Y(100);  //Select a point in the image to find the aberration at
    cv::Rect rect1(X-(isize/2), Y-(isize/2), isize, isize);   //Draw a rectangle centered at that point
    
    img = img(rect1).clone();
    cv::normalize(img, img, 0, 1, CV_MINMAX);
    std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;
  }
  
  int M = 14*2;
  TelescopeSettings ts(img.cols);
  //double data[] = { 0.0, 0.0, 0.0, 0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  double data[] = {   0.0, 0.0, 0.0, 0.3, 0.2, 0.7, 0.2, 0.4, 0.2, 0.5, 0.7, 0.5, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  cv::Mat coeffs(M, 1, cv::DataType<double>::type, data);
  cv::Mat d1, d2;
  aberrate(img, coeffs, ts.pupilRadiousPixels(), 0.0,    0.06, d1);
  aberrate(img, coeffs, ts.pupilRadiousPixels(), ts.k(), 0.06, d2);

  ImageQualityMetric iqm;
  cv::Mat d1_n;
  cv::normalize(d1, d1_n, 0, 1, CV_MINMAX);
  cv::Scalar mssimd1 = iqm.mssim(img, d1_n);
  std::cout << "MSSIM d1: " << mssimd1.val[0] << std::endl;
  
  cv::Mat d2_n;
  cv::normalize(d2, d2_n, 0, 1, CV_MINMAX);
  cv::Scalar mssimd2 = iqm.mssim(img, d2_n);
  std::cout << "MSSIM d2: " << mssimd2.val[0] << std::endl;
  
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};
  
  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d1);
  noiseDefocused.meanPowerSpectrum(d2);
  
  //double meanPower = (sigma.val[0]*sigma.val[0])/d1.total();
  std::vector<double> meanPowerNoise = {noiseFocused.meanPower(), noiseDefocused.meanPower()}; //{meanPower, meanPower};   //supposed same noise in both images
  cv::Mat object = wSensor.WavefrontSensing(d, meanPowerNoise);
  fftShift(object);
  cv::idft(object, object, cv::DFT_REAL_OUTPUT);
  cv::normalize(object, object, 0, 1, CV_MINMAX);
 
  cv::Scalar mssimIndex = iqm.mssim(img, object);
  std::cout << "MSSIM obj: " << mssimIndex.val[0] << std::endl;

}

void SubimageLayout::aberrate(const cv::Mat& img, const cv::Mat& aberationCoeffs, const double& pupilRadious, const double& diversity, const double& sigmaNoise, cv::Mat& aberratedImage)
{

  cv::Mat planes[] = {img, cv::Mat::zeros(img.size(), cv::DataType<double>::type)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(complexI);

  double pupilSideLength = img.cols;

  cv::Mat pupilAmplitude = Zernikes::phaseMapZernike(1, pupilSideLength, pupilRadious);
  
  cv::Mat z4 = Zernikes::phaseMapZernike(4, pupilSideLength, pupilRadious);
  double z4AtOrigen = z4.at<double>(z4.cols/2, z4.rows/2);
  cv::Mat diversityPhase = (diversity * 3.141592/(2.0*std::sqrt(3.0))) * (z4 - z4AtOrigen);

  cv::Mat pupilPhase = Zernikes::phaseMapZernikeSum(pupilSideLength, pupilRadious, aberationCoeffs);
  OpticalSystem OS = OpticalSystem(pupilPhase + diversityPhase, pupilAmplitude);  //Characterized optical system

  cv::Mat otf = OS.otf();
  fftShift(otf);

  cv::mulSpectrums(selectCentralROI(otf, complexI.size()), complexI.mul(complexI.total()), aberratedImage, cv::DFT_COMPLEX_OUTPUT);
  
  fftShift(aberratedImage);
  cv::idft(aberratedImage, aberratedImage, cv::DFT_REAL_OUTPUT);

  cv::Mat noise(img.size(), cv::DataType<double>::type);
  cv::Scalar sigma(sigmaNoise), m_(0);
  
  cv::theRNG() = cv::RNG( time (0) );
  cv::randn(noise, m_, sigma);
  
  cv::add(aberratedImage, noise, aberratedImage);
}


void SubimageLayout::navigateThrough()
{
  //Real dataset
  //nextSubimage()
  //The following actions should be done by the class in every sub-image
  cv::Mat dat1, dat2, img1, img2;
  readFITS("../inputs/sfoc.fits", dat1);
  readFITS("../inputs/sout.fits", dat2);
  dat1.convertTo(img1, cv::DataType<double>::type);
  dat2.convertTo(img2, cv::DataType<double>::type);

  std::cout << "cols: " << img1.cols << " x " << "rows: " << img1.rows << std::endl;

  long upperTopCorner = 26;
  subimageSize_ = 42; //70; //56; //size of the box to be analized
  cv::Rect rect(upperTopCorner, upperTopCorner, subimageSize_, subimageSize_);

  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(img1);
  noiseDefocused.meanPowerSpectrum(img2);
  
  cv::Mat d0 = img1(rect).clone();
  cv::Mat dk = img2(rect).clone(); 
  //writeFITS(d0, "../d0.fits");
  //writeFITS(dk, "../dk.fits");
  //cv::normalize(d0, d0);
  //cv::normalize(dk, dk);
  std::cout << "cv::mean(d0): " << cv::mean(d0).val[0] << std::endl;
  std::cout << "cv::mean(dk): " << cv::mean(dk).val[0] << std::endl;
  //d0 = d0-cv::mean(d0);
  //dk = dk-cv::mean(dk);
  //cv::normalize(d0, d0, 0, 1, CV_MINMAX);
  //cv::normalize(dk, dk, 0, 1, CV_MINMAX);
  
  std::cout << "noiseDefocused.sigma(): " << noiseDefocused.sigma() << std::endl;
  std::cout << "noiseFocused.sigma(): " << noiseFocused.sigma() << std::endl;

  //create apodization window and substract constant to have zero mean in the apodized image
  cv::Mat apodizationWindow;
  createModifiedHanningWindow(apodizationWindow, d0.cols, apodizationPercent_, cv::DataType<double>::type);
  cv::Scalar imageD0Offset = cv::sum(d0.mul(apodizationWindow))/cv::sum(apodizationWindow);
  cv::Scalar imageDkOffset = cv::sum(dk.mul(apodizationWindow))/cv::sum(apodizationWindow);

  WavefrontSensor wSensor;
  //std::vector<cv::Mat> d = {(d0-imageD0Offset).mul(apodizationWindow), (dk-imageDkOffset).mul(apodizationWindow)};
  std::vector<cv::Mat> d = {d0, dk};
  double sigma_d0 = ( cv::mean(d0).val[0] / 100.0 ) * 0.38;
  double sigma_dk = ( cv::mean(dk).val[0] / 100.0 ) * 0.33;
  std::cout << "sigma_d0: " << sigma_d0 << "; " << "sigma_dk: " << sigma_dk << std::endl;
  // {0.3, 0.4};
  std::vector<double> meanPowerNoise = {noiseFocused.meanPower(), noiseDefocused.meanPower()}; //{sigma_d0*sigma_d0/d0.total(), sigma_dk*sigma_dk/dk.total()};

  cv::Mat phaseResult = wSensor.WavefrontSensing(d, meanPowerNoise);
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
