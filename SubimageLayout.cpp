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
#include "ImageQualityMetric.h"
#include "PDTools.h"

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


void SubimageLayout::computerGeneratedImage()
{
  //Benchmark
  int isize = 128;
  cv::Mat img;
  if(true)
  {
    cv::Mat dat;
    readFITS("../inputs/surfi000.fits", dat);
    dat.convertTo(img, cv::DataType<double>::type);

    int X(100),Y(100);
    cv::Rect rect1(X, Y, isize, isize);
    
    img = img(rect1).clone();
    cv::normalize(img, img, 0, 1, CV_MINMAX);
    std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;
  }
  if(false)
  {
    img = cv::Mat::zeros(cv::Size(isize, isize), cv::DataType<double>::type);
    //by default int thickness=1, int lineType=8, int shift=0
    int quarter = isize/4;
    cv::circle(img, cv::Point(quarter,quarter), 10, cv::Scalar(1), -1);
    cv::circle(img, cv::Point(3*quarter,quarter), 10, cv::Scalar(1), -1);
    cv::circle(img, cv::Point(quarter,3*quarter), 10, cv::Scalar(1), -1);
    cv::circle(img, cv::Point(3*quarter,3*quarter), 10, cv::Scalar(1), -1);
    cv::normalize(img, img, 0, 1, CV_MINMAX);
  }
  
  cv::Mat planes[] = {img, cv::Mat::zeros(img.size(), cv::DataType<double>::type)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(complexI);
 
  double pupilRadious = 32.5011;
  double pupilSideLength = isize/2;

  int K = 2;
  int M = 14;
  /*
  double data[] = {-2.401125838894388e-16, 0.3522232233399166, -0.379281849524244, 0.3534864099022184, 0.208522857754222, 0.7264457570573712, 0.3846718779804479, 0.09719520962238018, -0.5976538896757668, 0.2002786634782341, -0.03975816621627151, 0.09184171169366809, -0.08375642972811849, 0.06625904289356371, 2.401125838894387e-16, -0.3522232233399165, 0.3792818495242439, 0.3534864099022182, 0.208522857754222, 0.7264457570573709, 0.3846718779804478, 0.09719520962238014, -0.5976538896757666, 0.200278663478234, -0.0397581662162715, 0.09184171169366806, -0.08375642972811846, 0.0662590428935637};
  */
  
  double data[] = {  0.0, 0.0, 0.0, 0.3, 0.2, 0.7, 0.2, 0.4, 0.2, 0.5, 0.7, 0.5, 0.1, 0.9,
                     0.0, 0.0, 0.0, 0.3, 0.2, 0.7, 0.2, 0.4, 0.2, 0.5, 0.7, 0.5, 0.1, 0.9};
  
  //double data[] = {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
  //                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 };
  
  cv::Mat coeffs(K*M, 1, cv::DataType<double>::type, data);
  
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, pupilRadious);
  
  ////////Consider the case of two diversity images
  std::vector<double> diversityFactor = {0.0, -2.21209};
  cv::Mat z4 = Zernikes<double>::phaseMapZernike(4, pupilSideLength, pupilRadious);
  double z4AtOrigen = Zernikes<double>::pointZernike(4, 0, 0);
  std::vector<cv::Mat> diversityPhase;
  for(double dfactor : diversityFactor)
  {
    //defocus zernike coefficient: c4 = dfactor * PI/(2.0*std::sqrt(3.0))
	  diversityPhase.push_back( (dfactor * 3.141592/(2.0*std::sqrt(3.0))) * (z4 - z4AtOrigen));
  }
  ////////
  
  std::vector<OpticalSystem> OS;
  for(int k=0; k<K; ++k)
  {  //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
    cv::Mat pupilPhase_i = Zernikes<double>::phaseMapZernikeSum(pupilSideLength,pupilRadious, coeffs(cv::Range(k*M, k*M + M), cv::Range::all()));
    OS.push_back(OpticalSystem(pupilPhase_i + diversityPhase.at(k), pupilAmplitude));  //Characterized optical system
  }
   
  cv::Mat d1, d2;
  cv::Mat otf1 = OS.front().otf();
  cv::Mat otf2 = OS.back().otf();
  fftShift(otf1);
  fftShift(otf2);

  cv::mulSpectrums(selectCentralROI(otf1, complexI.size()), complexI.mul(complexI.total()), d1, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(selectCentralROI(otf2, complexI.size()), complexI.mul(complexI.total()), d2, cv::DFT_COMPLEX_OUTPUT);

  fftShift(d1);
  fftShift(d2);
  cv::idft(d1, d1, cv::DFT_REAL_OUTPUT);
  cv::idft(d2, d2, cv::DFT_REAL_OUTPUT);
 
  cv::Mat noise1(isize, isize, cv::DataType<double>::type);
  cv::Mat noise2(isize, isize, cv::DataType<double>::type);
  cv::Scalar sigma(1.1), m_(0);
  
  cv::theRNG() = cv::RNG( time (0) );
  cv::randn(noise1, m_, sigma);
  
  cv::theRNG() = cv::RNG( time (0) );
  cv::randn(noise2, m_, sigma);

  cv::add(d1, noise1, d1);
  cv::add(d2, noise2, d2);

  ImageQualityMetric iqm;
  cv::Mat d1_n;
  cv::normalize(d1, d1_n, 0, 1, CV_MINMAX);
  cv::Scalar mssimd1 = iqm.mssim(img, d1_n);
  std::cout << "MSSIM d1: " << mssimd1.val[0] << std::endl;
  
  cv::Mat d2_n;
  cv::normalize(d2, d2_n, 0, 1, CV_MINMAX);
  cv::Scalar mssimd2 = iqm.mssim(img, d2_n);
  std::cout << "MSSIM d2: " << mssimd2.val[0] << std::endl;
  
  cv::Scalar imageD1Offset = cv::sum(d1)/cv::Scalar(d1.total());
  cv::Scalar imageD2Offset = cv::sum(d2)/cv::Scalar(d2.total());
  
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};
  double meanPower = (sigma.val[0]*sigma.val[0])/d1.total();
  std::vector<double> meanPowerNoise = {meanPower, meanPower};   //supposed same noise in both images
  cv::Mat object = wSensor.WavefrontSensing(d, meanPowerNoise);
  fftShift(object);
  cv::idft(object, object, cv::DFT_REAL_OUTPUT);
  cv::normalize(object, object, 0, 1, CV_MINMAX);
 
  //writeFITS(object, "../object.fits");
  //writeFITS(d1, "../d1.fits");
  //writeFITS(d2, "../d2.fits");
  //writeFITS(img, "../img.fits");
  
  cv::Scalar mssimIndex = iqm.mssim(img, object);
  std::cout << "MSSIM obj: " << mssimIndex.val[0] << std::endl;
  
}

void SubimageLayout::navigateThrough()
{
  //Real dataset
  //nextSubimage()
  //The following actions should be done by the class in every sub-image
  cv::Mat dat, img;
  readFITS("../inputs/pd.004.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);

  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  int X(100),Y(100);
  cv::Rect rect1(X, Y, subimageSize_, subimageSize_);
  cv::Rect rect2(936+X, 0+Y, subimageSize_, subimageSize_);

  cv::Mat d0 = img(rect1).clone();
  cv::Mat dk = img(rect2).clone();

  cv::Mat d0_norm;
  cv::normalize(d0, d0_norm, 0, 1, CV_MINMAX);
//  cv::imshow("d0", d0_norm);

  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d0);
  noiseDefocused.meanPowerSpectrum(dk);

  std::cout << "noiseDefocused.sigma(): " << noiseDefocused.sigma() << std::endl;
  std::cout << "noiseFocused.sigma(): " << noiseFocused.sigma() << std::endl;

  //create apodization window and substract constant to have zero mean in the apodized image
  cv::Mat apodizationWindow;
  createModifiedHanningWindow(apodizationWindow, d0.cols, apodizationPercent_, cv::DataType<double>::type);
  cv::Scalar imageD0Offset = cv::sum(d0.mul(apodizationWindow))/cv::sum(apodizationWindow);
  cv::Scalar imageDkOffset = cv::sum(dk.mul(apodizationWindow))/cv::sum(apodizationWindow);

  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {(d0-imageD0Offset).mul(apodizationWindow), (dk-imageDkOffset).mul(apodizationWindow)};
  std::vector<double> meanPowerNoise = {noiseFocused.meanPower(), noiseDefocused.meanPower()};
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
