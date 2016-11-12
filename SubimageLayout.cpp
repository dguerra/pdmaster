/*
 * SubimageLayout.cpp
 *
 *  Created on: Jan 27, 2014
 *      Author: dailos
 */

#include "SubimageLayout.h"
#include "WavefrontSensor.h"
#include "NoiseEstimator.h"
#include "Optics.h"
#include "FITS.h"
#include "Zernike.h"
#include "OpticalSetup.h"
#include "ImageQualityMetric.h"
#include "ToolBox.h"
#include "PhaseScreen.h"
//Rename as ImageSimulator o ImageFormation o ImageDispatcher


SubimageLayout::SubimageLayout()
{

}

SubimageLayout::~SubimageLayout()
{
  // TODO Auto-generated destructor stub
}


//Move this function to phasescreen
cv::Mat SubimageLayout::atmospheric_zernike_coeffs(const unsigned int& z_max, const double& D, const double& r0)
{
  //Build zernike covariance matrix
  //unsigned int nl(2);   //First zernike order to start with: nl = 2 means do not consider piston
  unsigned int nl(4);     //First zernike order to start with: nl = 4 means do not consider piston and tip/tilt
  cv::Mat_<double> zc(z_max - nl + 1, z_max - nl + 1);
  Zernike zrnk;
  for(unsigned int i = nl; i <= z_max; ++i)
  {
    for(unsigned int j = i; j <= z_max; ++j)
    {
      zc.at<double>(i - nl, j - nl) = zrnk.zernike_covar(i, j);
    }
  }
  cv::completeSymm(zc);
  cv::Mat eigenvalues, eigenvectors;
  cv::eigen(zc, eigenvalues, eigenvectors);
  
  cv::Mat_<double> b(z_max - nl + 1, 1);
  cv::Mat sqrt_eigenvalues;
  cv::sqrt(eigenvalues, sqrt_eigenvalues);
  //std::cout << "sqrt_eigenvalues: " << sqrt_eigenvalues.t() << std::endl;
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::randn(b, 0.0, 1.0);
  // D/r0 = 30 -> very strong turbulence conditions
  // D/r0 = 8 -> strong turbulence
  // D/r0 = 6 -> medium
  // D/r0 = 4 -> low
  cv::Mat z_coeffs = eigenvectors.t() * b.mul(sqrt_eigenvalues *  std::pow( D / r0, 5.0 / 6.0));
  //Add zernike order that haven't been considered in the covariance matrix (piston)
  cv::copyMakeBorder( z_coeffs, z_coeffs, nl - 1, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0.0) );
  
  return z_coeffs;
}

void SubimageLayout::dataSimulator()
{
  //Load ground thruth image and normalize
  cv::Mat img;
  readFITS("../inputs/surfi000.fits", img);
  img.convertTo(img, cv::DataType<double>::type);
  cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  //Initialize focused and defocused images
  cv::Mat d1 = cv::Mat::zeros(img.size(), img.type());
  cv::Mat d2 = cv::Mat::zeros(img.size(), img.type());

  //Set up optical configuration for this size of image
  OpticalSetup ts(img.cols);
  
  //set a value for noise variance
  double sigma_noise(0.0);

  //Genrate phase screen
  unsigned int nw = img.cols;   //Size of phase screen in pixels
  double r0 = 0.2 * (ts.pupilRadiousPixels()/4.2);   //Fred parameter in pixels
  double L0 = 10.0 * (ts.pupilRadiousPixels()/4.2);   //The outer scale of turbulence in pixels
  PhaseScreen phaseScreen;
  //cv::Mat phase = phaseScreen.fractalMethod(nw, r0, L0);
  
  double z_coeffs_d[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0};
  cv::Mat z_coeffs(14, 1, cv::DataType<double>::type, z_coeffs_d);
  Zernike zrnk;
  cv::Mat phase = zrnk.phaseMapZernikeSum(img.cols, ts.pupilRadiousPixels(), z_coeffs);
  
  aberrate(img, phase(cv::Rect(cv::Point(0,0),img.size())), ts.pupilRadiousPixels(),  sigma_noise, d1 );
  
  cv::Mat diversityPhase = (ts.k() * 3.141592/(2.0*std::sqrt(3.0))) *  zrnk.phaseMapZernike(4, img.cols, ts.pupilRadiousPixels(), false);
  aberrate(img, phase(cv::Rect(cv::Point(0,0),img.size())) + diversityPhase, ts.pupilRadiousPixels(), sigma_noise, d2 );
  
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};
  
  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d1);
  noiseDefocused.meanPowerSpectrum(d2);
  
  //double meanPower = (sigma.val[0]*sigma.val[0])/d1.total();

  //std::vector<double> meanPowerNoise = {0.0, 0.0};
  cv::Mat object = wSensor.WavefrontSensing(d, (noiseFocused.meanPower()+noiseDefocused.meanPower())/2.0);
}


void SubimageLayout::fromPhaseScreen()
{
  //Load ground thruth image and normalize
  cv::Mat img;
  readFITS("../inputs/surfi000.fits", img);
  img.convertTo(img, cv::DataType<double>::type);
  cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  //Initialize focused and defocused images
  cv::Mat d1 = cv::Mat::zeros(img.size(), img.type());
  cv::Mat d2 = cv::Mat::zeros(img.size(), img.type());
  
  //Set up optical configuration for this size of image
  OpticalSetup ts(img.cols);
  
  //set a value for noise variance
  double sigma_noise(0.0);
  
  //load phase screen from disc and applyOpticalAberration
  cv::Mat pupilPhase;
  readFITS("../inputs/wavefront0001pupil.fits", pupilPhase);
  pupilPhase.convertTo(pupilPhase, cv::DataType<double>::type);
  cv::subtract(pupilPhase, 66417.0, pupilPhase);
  
  // specify fx and fy and let the function compute the destination image size.
  cv::resize(pupilPhase, pupilPhase, cv::Size(), ts.pupilRadiousPixels()/455.0, ts.pupilRadiousPixels()/455.0, cv::INTER_LINEAR);
  
  
  int top  = (int) ((img.rows-pupilPhase.rows)/2.0); int bottom = (img.rows - pupilPhase.rows) - top;
  int left = (int) ((img.cols-pupilPhase.cols)/2.0); int right  = (img.cols - pupilPhase.cols) - left;
  cv::copyMakeBorder( pupilPhase, pupilPhase, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0.0) );
  
  std::cout << "pupilPhase.cols: " << pupilPhase.cols << " x " << "pupilPhase.rows: " << pupilPhase.rows << std::endl;
  aberrate(img, pupilPhase, ts.pupilRadiousPixels(),  sigma_noise, d1 );
  Zernike zrnk;
  cv::Mat diversityPhase = (ts.k() * 3.141592/(2.0*std::sqrt(3.0))) *  zrnk.phaseMapZernike(4, img.cols, ts.pupilRadiousPixels(), false);
  aberrate(img, pupilPhase + diversityPhase, ts.pupilRadiousPixels(), sigma_noise, d2 );
/*
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};

  std::vector<double> meanPowerNoise = {0.0, 0.0};
  cv::Mat object = wSensor.WavefrontSensing(d, meanPowerNoise);
*/
}

//Rename as simulation image
void SubimageLayout::computerGeneratedImage()
{
  //Read ground truth image from fits file
  cv::Mat img, dat;
  readFITS("../inputs/surfi000.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);
  cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;
  
/*  
  //transfor to fourier domain and brings energy to the center
  cv::Mat D;
  cv::dft(img, D, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(D);
  //remove frequencies beyond cutoff
  OpticalSetup tsettings(D.cols);
  D.setTo(0, Zernike::phaseMapZernike(1, D.cols, tsettings.cutoffPixel()) == 0);
  //Take back to image domain the remaing frequencies
  fftShift(D);
  cv::idft(D, img, cv::DFT_REAL_OUTPUT);
*/

  //Draw subimage layout
  int tileSize = 34;
  unsigned int pixelsBetweenTiles = (int)(img.cols);
  std::vector<cv::Mat> img_v;
  std::vector<std::pair<cv::Range, cv::Range> > rng_v;
  divideIntoTiles(img.size(), pixelsBetweenTiles, tileSize, rng_v);
  for(auto rng_i : rng_v) img_v.push_back( img(rng_i.first, rng_i.second).clone() );
  OpticalSetup ts(tileSize);  
  std::vector<cv::Mat> phase_v;
  
  
  //Create at least one phase for one patch
  double data_coeffs[] =   {0, 0, 0,  0.2155518876905822, -0.1944677950837682, 0.03497835759983991, -0.1114719556999538, -0.0089693894577957,
          
          -0.04710748628638275,  0.1028641408486822, -0.0390145418007589,   0.05894036261075137,  0.0756139441983438, -0.0645236207777915, 
         	-0.01968917879771064, -0.0391565561963278,  0.0198497982483514,  -0.02280286747790469, -0.0564022395673681, -0.0148990812317611};
         
  cv::Mat coeffs(sizeof(data_coeffs)/sizeof(*data_coeffs), 1, cv::DataType<double>::type, data_coeffs);
  Zernike zrnk;
  cv::Mat phase = zrnk.phaseMapZernikeSum(img_v.front().cols, ts.pupilRadiousPixels(), coeffs);
  phase_v.push_back(phase);
  //Create extra phase to be added to zernike coefficient number four
  cv::Mat extraZ4 = (ts.k() * 3.141592/(2.0*std::sqrt(3.0))) *  zrnk.phaseMapZernike(4, img_v.front().cols, ts.pupilRadiousPixels());
  
  
  /* //Apodize image to avoid edge effects
  cv::Mat hannWindow;
  createModifiedHanningWindow(hannWindow, tileSize, 20.0, cv::DataType<double>::type);
  cv::Scalar sum_hann = cv::sum(hannWindow);
  cv::Scalar offset_d1 = cv::sum(d1.mul(hannWindow))/sum_hann;
  D1 = fft(( d1 - offset_d1) * hannWindow)
  */
  
  //Focused and defocused images
  cv::Mat d1 = cv::Mat::zeros(img.size(), img.type());
  cv::Mat d2 = cv::Mat::zeros(img.size(), img.type());
  
  double sigma_noise(0.0);
  
  for(unsigned int i=0;i<img_v.size(); ++i)
  {
    cv::Mat tile1, tile2;
    std::pair<cv::Range, cv::Range> rng = rng_v.at(i);
    
    aberrate(img_v.at(i), phase_v.at(i), ts.pupilRadiousPixels(),  sigma_noise, tile1 );
    aberrate(img_v.at(i), phase_v.at(i) + extraZ4, ts.pupilRadiousPixels(), sigma_noise, tile2 );
    
    tile1.copyTo( d1(rng.first, rng.second) );
    tile2.copyTo( d2(rng.first, rng.second) );
    
  }
  
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};
  
  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d1);
  noiseDefocused.meanPowerSpectrum(d2);
  
  std::cout << "noiseFocused.sigma: " << noiseFocused.sigma() << std::endl;
  std::cout << "noiseDefocused.sigma: " << noiseDefocused.sigma() << std::endl;
  cv::Mat object = wSensor.WavefrontSensing(d, (noiseFocused.meanPower()+noiseDefocused.meanPower())/2.0);
}

void SubimageLayout::aberrate(const cv::Mat& img, const cv::Mat& aberrationPhase, const double& pupilRadious, const double& sigmaNoise, cv::Mat& aberratedImage)
{

  cv::Mat planes[] = {img, cv::Mat::zeros(img.size(), cv::DataType<double>::type)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(complexI);

  double pupilSideLength = img.cols;

//  cv::Mat diversityPhase = (diversity * 3.141592/(2.0*std::sqrt(3.0))) * Zernike::phaseMapZernike(4, pupilSideLength, pupilRadious, false);
//  cv::Mat pupilPhase = Zernike::phaseMapZernikeSum(pupilSideLength, pupilRadious, aberationCoeffs);
  cv::Mat c_mask;
  Zernike zrnk;
  zrnk.circular_mask(pupilRadious, pupilSideLength, c_mask);
  Optics OS = Optics(aberrationPhase, c_mask );  //Characterized optical system

  cv::Mat otf = OS.otf();
  fftShift(otf);

  cv::mulSpectrums(selectCentralROI(otf, complexI.size()), complexI.mul(complexI.total()), aberratedImage, cv::DFT_COMPLEX_OUTPUT);
  
  fftShift(aberratedImage);
  cv::idft(aberratedImage, aberratedImage, cv::DFT_REAL_OUTPUT);


  //Add noise to the image
  cv::Mat noise(img.size(), cv::DataType<double>::type);
  cv::Scalar sigma(sigmaNoise), m_(0);
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::randn(noise, m_, sigma);
  cv::add(aberratedImage, noise, aberratedImage);
  
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

