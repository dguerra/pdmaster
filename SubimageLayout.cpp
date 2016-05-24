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
#include "BasisRepresentation.h"
#include "OpticalSetup.h"
#include "ImageQualityMetric.h"
#include "ToolBox.h"
//Rename as ImageSimulator o ImageFormation o ImageDispatcher


SubimageLayout::SubimageLayout()
{

}

SubimageLayout::~SubimageLayout()
{
  // TODO Auto-generated destructor stub
}

cv::Mat SubimageLayout::atmospheric_zernike_coeffs(const unsigned int& z_max, const double& D, const double& r0)
{
  //Build zernike covariance matrix
  //unsigned int nl(2);   //First zernike order to start with: nl = 2 means do not consider piston
  unsigned int nl(4);     //First zernike order to start with: nl = 4 means do not consider piston and tip/tilt
  cv::Mat_<double> zc(z_max - nl + 1, z_max - nl + 1);
  
  for(unsigned int i = nl; i <= z_max; ++i)
  {
    for(unsigned int j = i; j <= z_max; ++j)
    {
      zc.at<double>(i - nl, j - nl) = BasisRepresentation::zernike_covar(i, j);
    }
  }
  cv::completeSymm(zc);
  cv::Mat eigenvalues, eigenvectors;
  cv::eigen(zc, eigenvalues, eigenvectors);
  
  cv::Mat_<double> b(z_max - nl + 1, 1);
  cv::Mat sqrt_eigenvalues;
  cv::sqrt(eigenvalues, sqrt_eigenvalues);
  std::cout << "sqrt_eigenvalues: " << sqrt_eigenvalues.t() << std::endl;
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::randn(b, 0.0, 1.0);
  cv::Mat z_coeffs = eigenvectors.t() * b.mul(sqrt_eigenvalues *  std::pow( D / r0, 5.0 / 6.0));
  //Add zernike order that haven't been considered in the covariance matrix (piston)
  cv::copyMakeBorder( z_coeffs, z_coeffs, nl - 1, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0.0) );
  
  return z_coeffs;
}

void SubimageLayout::realData()
{
  cv::Mat img_f, dat_f;
  readFITS("../inputs/sfoc.fits", dat_f);
  dat_f.convertTo(img_f, cv::DataType<double>::type);
  cv::normalize(img_f, img_f, 0, 1, CV_MINMAX);
  //writeFITS(img_f, "../img_f.fits");
  
  cv::Mat img_d, dat_d;
  readFITS("../inputs/sout.fits", dat_d);
  dat_d.convertTo(img_d, cv::DataType<double>::type);
  cv::normalize(img_d, img_d, 0, 1, CV_MINMAX);
  //writeFITS(img_d, "../img_d.fits");
 
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {img_f, img_d};
  
  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(img_f);
  noiseDefocused.meanPowerSpectrum(img_d);
  
  std::vector<double> meanPowerNoise = {noiseFocused.meanPower(), noiseDefocused.meanPower()}; //{meanPower, meanPower};   //supposed same noise in both images
  cv::Mat object = wSensor.WavefrontSensing(d, meanPowerNoise); 
}

//Rename as simulation image
void SubimageLayout::computerGeneratedImage()
{
  //Benchmark
  cv::Mat img, dat;
  readFITS("../inputs/surfi000.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);
  cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
  //writeFITS(img, "../img.fits");
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  //int M = 14;
  
  /*
  double data1[] =   { 0.0, 0.0, 0.0, 0.33, 0.21, 0.79, 0.22, 0.44, 0.26, 0.59, 0.79, 0.54, 0.12, 0.99}; 
  double data2[] =   { 0.0, 0.0, 0.0, 0.34, 0.22, 0.78, 0.23, 0.45, 0.24, 0.58, 0.79, 0.55, 0.12, 0.99}; 
  double data3[] =   { 0.0, 0.0, 0.0, 0.35, 0.23, 0.77, 0.21, 0.46, 0.24, 0.58, 0.77, 0.55, 0.12, 0.98}; 
  double data4[] =   { 0.0, 0.0, 0.0, 0.34, 0.21, 0.77, 0.22, 0.46, 0.25, 0.57, 0.78, 0.54, 0.11, 0.98}; 
  */
  
  //high Sparsity case:
  double data1[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.79, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54, 0.0, 0.0 };
  double data2[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0, 0.0 };
  double data3[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0, 0.0 };
  double data4[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54, 0.0, 0.0 };
  
  //enum class Dataset = {RealDataImage, AtmosphericTurbulence, RandomSparseCoeffs};
  
  
  /*
  cv::Mat coeffs1(M, 1, cv::DataType<double>::type, data1);
  cv::Mat coeffs2(M, 1, cv::DataType<double>::type, data2);
  cv::Mat coeffs3(M, 1, cv::DataType<double>::type, data3);
  cv::Mat coeffs4(M, 1, cv::DataType<double>::type, data4);
  */
  //std::cout << "atmospheric_zernike_coeffs(max_zernike, 1.0, 1.0): " << atmospheric_zernike_coeffs(200, 1.0, 1.0).t() << std::endl;
  double fc = 1.0;
  unsigned int max_zernike(50);
  cv::Mat coeffs1 = fc * atmospheric_zernike_coeffs(max_zernike, 1.0, 1.0);
  cv::Mat coeffs2 = fc * atmospheric_zernike_coeffs(max_zernike, 1.0, 1.0);
  cv::Mat coeffs3 = fc * atmospheric_zernike_coeffs(max_zernike, 1.0, 1.0);
  cv::Mat coeffs4 = fc * atmospheric_zernike_coeffs(max_zernike, 1.0, 1.0);
  
  std::cout << "coeffs1: " << coeffs1.t() << std::endl;
  std::cout << "coeffs2: " << coeffs2.t() << std::endl;
  std::cout << "coeffs3: " << coeffs3.t() << std::endl;
  std::cout << "coeffs4: " << coeffs4.t() << std::endl;
  
  std::vector<cv::Mat> coeffs_v = {coeffs1, coeffs2, coeffs3, coeffs4};
  unsigned int pixelsBetweenTiles = (int)(img.cols);
  int tileSize = 34;
  std::vector<cv::Mat> img_v;
  std::vector<std::pair<cv::Range, cv::Range> > rng_v;
  divideIntoTiles(img.size(), pixelsBetweenTiles, tileSize, rng_v);
  for(auto rng_i : rng_v) img_v.push_back( img(rng_i.first, rng_i.second).clone() );
  
  
  cv::Mat d1 = cv::Mat::zeros(img.size(), img.type());
  cv::Mat d2 = cv::Mat::zeros(img.size(), img.type());
  
  OpticalSetup ts(tileSize);
  double sigma_noise(0.01);

  for(unsigned int i=0;i<img_v.size(); ++i)
  {
    cv::Mat tile1, tile2;
    std::pair<cv::Range, cv::Range> rng = rng_v.at(i);
    cv::Mat pupilPhase = BasisRepresentation::phaseMapZernikeSum(img_v.at(i).cols, ts.pupilRadiousPixels(), coeffs_v.at(i));
    aberrate(img_v.at(i), pupilPhase, ts.pupilRadiousPixels(),  sigma_noise, tile1 );

    cv::Mat diversityPhase = (ts.k() * 3.141592/(2.0*std::sqrt(3.0))) * BasisRepresentation::phaseMapZernike(4, img_v.at(i).cols, ts.pupilRadiousPixels(), false);
    aberrate(img_v.at(i), pupilPhase + diversityPhase, ts.pupilRadiousPixels(), sigma_noise, tile2 );
    tile1.copyTo( d1(rng.first, rng.second) );
    tile2.copyTo( d2(rng.first, rng.second) );
    
  }
  //writeFITS(d2, "../tile2.fits");
  
  WavefrontSensor wSensor;
  std::vector<cv::Mat> d = {d1, d2};
  
  NoiseEstimator noiseFocused, noiseDefocused;
  noiseFocused.meanPowerSpectrum(d1);
  noiseDefocused.meanPowerSpectrum(d2);
  
  //double meanPower = (sigma.val[0]*sigma.val[0])/d1.total();
  std::vector<double> meanPowerNoise = {noiseFocused.meanPower(), noiseDefocused.meanPower()}; //{meanPower, meanPower};   //supposed same noise in both images
  //std::vector<double> meanPowerNoise = {0.0, 0.0};
  cv::Mat object = wSensor.WavefrontSensing(d, meanPowerNoise);
}

void SubimageLayout::aberrate(const cv::Mat& img, const cv::Mat& aberrationPhase, const double& pupilRadious, const double& sigmaNoise, cv::Mat& aberratedImage)
{

  cv::Mat planes[] = {img, cv::Mat::zeros(img.size(), cv::DataType<double>::type)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(complexI);

  double pupilSideLength = img.cols;

//  cv::Mat diversityPhase = (diversity * 3.141592/(2.0*std::sqrt(3.0))) * BasisRepresentation::phaseMapZernike(4, pupilSideLength, pupilRadious, false);
//  cv::Mat pupilPhase = BasisRepresentation::phaseMapZernikeSum(pupilSideLength, pupilRadious, aberationCoeffs);
  Optics OS = Optics(aberrationPhase, BasisRepresentation::circular_mask(pupilRadious, pupilSideLength) );  //Characterized optical system

  cv::Mat otf = OS.otf();
  fftShift(otf);

  cv::mulSpectrums(selectCentralROI(otf, complexI.size()), complexI.mul(complexI.total()), aberratedImage, cv::DFT_COMPLEX_OUTPUT);
  
  fftShift(aberratedImage);
  cv::idft(aberratedImage, aberratedImage, cv::DFT_REAL_OUTPUT);

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

/*
//phaseScreen:
cv::Mat phasescreen(const unsigned int& nw, const double& r0, const double& L0)
{
  // w=phasescreen(nw,r0,L0)
  //   D(r) = 6.88*(r/r0)^(5/3)   r<L0
  //   D(r) = 6.88*(L0/r0)^(5/3)  r>=l0
  //   sigma_w = sqrt(1/2 D(L0))
  //
  
  auto Dw = [](const double& r, const double& r0, const double& L0) -> double { return 6.88 * std::pow(std::min(r,L0) / r0, 5/3); };
  
  auto Cw = [](const double& r, const double& r0, const double& L0) -> double { return 3.44 * std::pow(L0/r0, 5/3) - 3.44 * std::pow(std::min(r,L0)/r0, 5/3); };
  
  double np = std::ceil(std::log(nw - 1.0)/std::log(2.0));
  double n  = std::pow(2.0, np) + 1;
  cv::Mat w = cv::Mat::zeros(n, n, cv::DataType<double>::type);
  
  double wrms2 = 0.5 * Dw(L0,r0,L0);
  
  double c0 = wrms2;
  double r = std::min(n-1, L0);
  double c1 = wrms2 - 0.5 * Dw(n-1, r0, L0);
  r = std::min(std::sqrt(2.0) * (n - 1.0), L0);
  double c2 = wrms2 - 0.5 * Dw(std::sqrt(2.0) * (n - 1.0), r0, L0);
  
  double data_C[] = { c0, c1, c1, c2,
                      c1, c0, c2, c1,
                      c1, c2, c0, c1,
                      c2, c1, c1, c0 }; // Covariance matrix
                      
  cv::Mat C(4, 4, cv::DataType<double>::type, data_C);
                     
  K=chol(C,'lower');  //  C=K*K'
  
  u=K*randn(4,1);
  w(1,1)=u(1);
  w(n,1)=u(2);
  w(1,n)=u(3);
  w(n,n)=u(4);
  
  for il=1:np
  for(unsigned int il = 1; il<np; ++il)
  {
      int d = std::pow(2.0, np - il + 1);  // Lado de la celda
      double alfa  = Cw((double)d / std::sqrt(2.0), r0, L0) / (wrms2 + 2.0 * Cw(d,r0,L0) + Cw(std::sqrt(2.0) * d, r0, L0));
      double alfa0 = std::sqrt(wrms2 - 4.0 * alfa*alfa * (wrms2 + 2.0 * Cw(d,r0,L0) + Cw(std::sqrt(2.0) * d,r0,L0)));
      p=(d/2+1):d:n;
      q=(d+1):d:(n-d);
      w(p,p)=alfa0*randn(length(p))+alfa*(w(p+d/2,p+d/2)+w(p+d/2,p-d/2)+w(p-d/2,p-d/2)+w(p-d/2,p+d/2));
      if il>1
          alfa=Cw(d/2,r0,L0)/(wrms2+2*Cw(d/sqrt(2),r0,L0)+Cw(d,r0,L0));
          alfa0=sqrt(wrms2-4*alfa^2*(wrms2+2*Cw(d/sqrt(2),r0,L0)+Cw(d,r0,L0)));
          w(p,q)=alfa0*randn(length(p),length(q))+alfa*(w(p+d/2,q)+w(p-d/2,q)+w(p,q+d/2)+w(p,q-d/2));
          w(q,p)=alfa0*randn(length(q),length(p))+alfa*(w(q+d/2,p)+w(q-d/2,p)+w(q,p+d/2)+w(q,p-d/2));
      end
      A(1,1)=wrms2+Cw(d,r0,L0);
      A(1,2)=Cw(d/sqrt(2),r0,L0);
      A(2,1)=2*Cw(d/sqrt(2),r0,L0);
      A(2,2)=wrms2;
      alfa=A\[1;1]*Cw(d/2,r0,L0);
      alfa0=sqrt(wrms2-2*alfa(1)^2*(wrms2+Cw(d,r0,L0))-alfa(2)^2*wrms2-4*alfa(1)*alfa(2)*Cw(d/sqrt(2),r0,L0));
      w(p,1)=alfa0*randn(length(p),1)+alfa(1)*(w(p+d/2,1)+w(p-d/2,1))+alfa(2)*w(p,1+d/2);
      w(p,n)=alfa0*randn(length(p),1)+alfa(1)*(w(p+d/2,n)+w(p-d/2,n))+alfa(2)*w(p,n-d/2);
      w(1,p)=alfa0*randn(1,length(p))+alfa(1)*(w(1,p+d/2)+w(1,p-d/2))+alfa(2)*w(1+d/2,p);
      w(n,p)=alfa0*randn(1,length(p))+alfa(1)*(w(n,p+d/2)+w(n,p-d/2))+alfa(2)*w(n-d/2,p);
  }
  
}
*/