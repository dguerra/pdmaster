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
#include <fstream>
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
  
  cv::Mat img;
  //readFITS("../inputs/surfi000.fits", img);
  img = cv::imread(std::string("../inputs/vray_osl_simplex_noise_10.png"), CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

  img.convertTo(img, CV_64F);   //Convert to float 32 which is the data type used by the convNet
  cv::normalize(img, img, 0.0, 100.0, CV_MINMAX);     //¿¿?? Normalize before or after aberration
  
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  //Number of zernike polinomials used in the simulation: 10
  //Atmosphere coefficients vary randomly
  constexpr unsigned int numberOfZrnks(10);
  
   //The phase diversity coefficients define the different beam path in the optical setup
  cv::Matx<double, numberOfZrnks, 1> phase_div_0(0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  cv::Matx<double, numberOfZrnks, 1> phase_div_1(0.0, 0.0, 0.0,  2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  cv::Matx<double, numberOfZrnks, 1> phase_div_2(0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  std::vector<cv::Mat> phase_div = {cv::Mat(phase_div_0), cv::Mat(phase_div_1), cv::Mat(phase_div_2)};
  
  unsigned int tileSize(64);  //Size of subimage
  //We use the optical configuration parameters to find the radious of the aperture in pixels
  OpticalSetup ts(tileSize);

  //set a value for noise variance
  double sigma_noise(0.0);    //Add zero noise initially
  Zernike zrnk(ts.pupilRadiousPixels(), tileSize, numberOfZrnks);
  cv::Mat c_mask;
  zrnk.circular_mask(c_mask);
  
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::RNG& rng = cv::theRNG();
  
  const unsigned int stride(1);
  unsigned int batch_i(1);
  std::ofstream fout;
  
    //theBeginning:
  unsigned int countRecord(1);
  for(;;)  
  {
    //Rotate 90
    cv::transpose(img, img);  
    cv::flip(img, img,1);
    for (unsigned int r = 0; r < img.rows; r += stride)
    {
      for (unsigned int c = 0; c < img.cols; c += stride)
      {
        if( (r + tileSize) <= img.rows && (c + tileSize) <= img.cols)   //Only consider square tiles and discard the rest
        {
          //Get ready the file to be written
          if(!fout.is_open())
          {
            std::string filemane = std::string("../data_batch_") + std::to_string(batch_i) + std::string(".bin");
            fout.open(filemane, std::ios::out | std::ios::binary);
            if(!fout) {
              std::cout << "Cannot open file.";
              return;
            }
            countRecord = 1;  //Reset counter to one
          }
          
          std::vector<cv::Mat> v;
          cv::Matx<double, numberOfZrnks, 1> atmos(0.0, 0.0, 0.0, rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)
                                                                , rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)
                                                                , rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)
                                                                , rng.uniform(-20.0, 20.0) );
          cv::Mat atmos_;
          cv::Mat(atmos).convertTo(atmos_, CV_32F);
          fout.write(reinterpret_cast<char*>(atmos_.data), atmos_.total() * atmos_.elemSize());
         
          cv::Range rRange(r, r+tileSize), cRange(c, c+tileSize);
          for(cv::Mat pd : phase_div)
          {
            
            cv::Mat phase;
            zrnk.synthesize(cv::Mat(atmos) + pd, phase); 
            Optics optics(phase, c_mask);  //Characterized optical system
            cv::Mat otf = optics.otf().clone();
             
            cv::Mat psf;
            cv::idft(otf, psf, cv::DFT_REAL_OUTPUT);
            fftShift(psf);
            
            cv::Mat d_i;   //Data image captured by each detector
            bool correlation(false), full(false);
            convolveDFT(img(rRange, cRange), psf, d_i, correlation, full);
            d_i.convertTo(d_i, CV_32F);
            //Add noise if needed
            
            if(false)
            {
              //Visualize
              cv::Mat otf_[2];
              cv::split(otf, otf_);
              cv::magnitude(otf_[0], otf_[1], otf_[0]);
              cv::normalize(otf_[0], otf_[0], 0.0, 255.0, CV_MINMAX);
              cv::imwrite("../otf_mag.jpg", otf_[0]);
              
              cv::Mat psf_ = psf.clone();
              //cv::log(psf_, psf_);
              cv::normalize(psf_, psf_, 0.0, 255.0, CV_MINMAX);
              cv::imwrite("../psf.jpg", psf_);
    
              v.push_back(d_i);
            }
            
            fout.write(reinterpret_cast<char*>(d_i.data), d_i.total() * d_i.elemSize());
          }
          
          //In order BGR
          //cv::Mat img_color;
          //cv::merge(v, img_color);
          //cv::normalize(img_color, img_color, 0.0, 255.0, CV_MINMAX);
          //cv::imwrite("../img_color.jpg", img_color);
          if(countRecord == 10000)
          {
            fout.close(); 
            std::cout << "Batch created." << std::endl; 
            if(batch_i<10) batch_i++;
            else return;
          }
          
          countRecord++;
        }
      }
    }
  }
  
  std::cout << "End of function has been reached." << std::endl;
  //fout.close();
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
  cv::Mat c_mask;
  zrnk.circular_mask(ts.pupilRadiousPixels(), tileSize, c_mask);
  
  for(unsigned int i=0;i<img_v.size(); ++i)
  {
    cv::Mat tile1, tile2;
    std::pair<cv::Range, cv::Range> rng = rng_v.at(i);

    /*
    bool correlation(false), full(false);
    
    Optics optics_1(phase_v.at(i), c_mask);  //Characterized optical system
    cv::Mat psf1;
    cv::idft(optics_1.otf(), psf1, cv::DFT_REAL_OUTPUT);
    convolveDFT(img_v.at(i), psf1, tile1, correlation, full);

    Optics optics_2(phase_v.at(i) + extraZ4, c_mask);  //Characterized optical system
    cv::Mat psf2;
    cv::idft(optics_2.otf(), psf2, cv::DFT_REAL_OUTPUT);
    convolveDFT(img_v.at(i), psf2, tile2, correlation, full);
    */
    
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
  cv::Mat complexI;
  cv::dft(img, complexI, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(complexI);

  double pupilSideLength = img.cols;

  cv::Mat c_mask;
  Zernike zrnk;
  zrnk.circular_mask(pupilRadious, pupilSideLength, c_mask);
  Optics OS = Optics(aberrationPhase, c_mask );  //Characterized optical system
  
  cv::Mat otf = OS.otf();
  fftShift(otf);
  cv::mulSpectrums(otf, complexI.mul(complexI.total()), aberratedImage, cv::DFT_COMPLEX_OUTPUT);
  
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

