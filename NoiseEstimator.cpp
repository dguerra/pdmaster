#include "ToolBox.h"
#include "CustomException.h"

#include <iostream>

#include "NoiseEstimator.h"
#include "OpticalSetup.h"
#include "Zernike.h"
#include "FITS.h"

NoiseEstimator::NoiseEstimator()
{
  sigma_ = 0;
  sigma2_ = 0;
  meanPower_ = 0;
}

NoiseEstimator::~NoiseEstimator()
{
  // TODO Auto-generated destructor stub
}

//The method could be enhanced with multiresolution support
void NoiseEstimator::kSigmaClipping(const cv::Mat& img)
{
   cv::Mat imgBlurred(img.size(), img.type());
   cv::Mat mask(img.size(), CV_8U); //mask has to be this type CV_8U
   mask = cv::Scalar::all(255);
   int blurKernelSize(3), k(3);
   cv::medianBlur(img, imgBlurred, blurKernelSize);
   cv::Mat imgNoise(img.size(), CV_32F);
   imgNoise = img - imgBlurred;
   cv::Scalar m, s;
   for(unsigned int l(0);l<3;++l)
   {
     cv::meanStdDev(imgNoise, m, s, mask);
     sigma_ = s.val[0];
     //Discard pixels greater or lower than 3*sigma and start all over again
     //cv::threshold(imgNoise, imgNoise, 0.0, 255.0, 0);
     //CAUTION: check sigma definition for meanStdDev! might not be the expected
     inRange(imgNoise, cv::Scalar(-k*sigma_), cv::Scalar(k*sigma_), mask);
   }
}

void NoiseEstimator::meanPowerSpectrum(const cv::Mat& img)
{
  if (img.channels() == 1)
  {
    cv::Mat IMG;
    //CAUTION: The result of the dft is explicitly scaled, so the sigma value might change
    cv::dft(img, IMG, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    fftShift(IMG);
    bool conjB(true);
  
    cv::Mat powerSpectrum;
    cv::mulSpectrums(IMG, IMG, powerSpectrum, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE, conjB);
    //A mask has to be applied to before calculating the sigma
    OpticalSetup tsettings(img.cols);
    std::cout << "cutoffPixel: " << tsettings.cutoffPixel() << std::endl;
    cv::Mat mask = cv::Mat::ones(powerSpectrum.size(), powerSpectrum.depth());

    //Mask to only consider pixel values after cutoff, which are due to noise only
    Zernike zrnk;
    mask.setTo(0, zrnk.phaseMapZernike(1, mask.cols, tsettings.cutoffPixel()) != 0);
    //mask.setTo(0, zrnk.phaseMapZernike(1, mask.cols, mask.cols/2) != 0);
    mask.colRange(int((mask.cols/2)-(mask.cols/6)),int((mask.cols/2)+(mask.cols/6))) = cv::Scalar(0);
    mask.rowRange(int((mask.rows/2)-(mask.rows/6)),int((mask.rows/2)+(mask.rows/6))) = cv::Scalar(0);

    //CAUTION: Remember to shift quadrants before applying circle mask to fourier domain!!
    cv::Mat croppedSpectrums = (splitComplex(powerSpectrum).first).mul(mask);
    meanPower_ = cv::sum(croppedSpectrums).val[0]/cv::sum(mask).val[0];
    sigma2_ = meanPower_ * img.total() ;
    sigma_ = std::sqrt(sigma2_);
    //sigma_ = img.cols * std::sqrt(meanPower_); //cd m                     ;(Append.K) 
    std::cout << "cv::mean(img): " << cv::mean(img) << std::endl;
    std::cout << "sigma = " << sigma_ << " represents " << sigma_/(cv::mean(img)/100.0) << "% of the average intensity." << std::endl;
  }
  else
  {
    throw CustomException("meanPowerSpectrum: Only one channel image allowed.");
  }
}


