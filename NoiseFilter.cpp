/*
 * NoiseFilter.cpp
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */

#include "NoiseFilter.h"
#include "ToolBox.h"
#include "OpticalSetup.h"
#include "BasisRepresentation.h"


NoiseFilter::NoiseFilter()
{
  // TODO Auto-generated constructor stub

}

NoiseFilter::~NoiseFilter()
{
  // TODO Auto-generated destructor stub
}

NoiseFilter::NoiseFilter(const cv::Mat& T0, const cv::Mat& Tk, const cv::Mat& D0, const cv::Mat& Dk,
                         const cv::Mat& Q2, const double& meanPowerNoiseD0, const double& meanPowerNoiseDk)
{
  const double filter_upper_limit(1.0);
  const double filter_lower_limit(0.1);

  cv::Mat filterDenomimator, smoothedFilterDenomimator;
  cv::Mat D0T0, DkTk;
  cv::mulSpectrums(D0,conjComplex(T0), D0T0, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(Dk,conjComplex(Tk), DkTk, cv::DFT_COMPLEX_OUTPUT);
  //cv::Mat absTerm = absComplex(D0T0 + ((meanPowerNoiseD0/meanPowerNoiseDk)*DkTk));   //we don't use this version anymore
  cv::Mat absTerm = absComplex(D0T0 + DkTk);
  
  //Both Q2 and absTerm should be single channel images (real images)
  cv::multiply(Q2, absTerm.mul(absTerm), filterDenomimator);
  
  cv::blur(filterDenomimator, smoothedFilterDenomimator, cv::Size(3,3));

  //smoothedFilterDenomimator.setTo(1.0e-35, smoothedFilterDenomimator < 1.0e-35);  //CAUTION! I need an explanation!
  cv::pow(smoothedFilterDenomimator, -1.0, smoothedFilterDenomimator);
  
  cv::Mat filterH = meanPowerNoiseD0 * smoothedFilterDenomimator;
  
  cv::Mat filterHFlipped;
  cv::flip(filterH, filterHFlipped, -1); //flipCode => -1 < 0 means two axes flip
  shift(filterHFlipped, filterHFlipped, 1, 1);  //shift matrix => 1 means one pixel to the right

  //corrects deviations from even function and substract from 1
  cv::Mat H = 1.0 - ((filterH + filterHFlipped)/2.0);
  
  H.setTo(0, H < filter_lower_limit);
  H.setTo(filter_upper_limit, H > filter_upper_limit);

  //To zero-out frequencies beyond cutoff
  OpticalSetup tsettings(T0.cols);
  H.setTo(0, BasisRepresentation::phaseMapZernike(1, H.cols, tsettings.cutoffPixel()) == 0);
  
  //select only the central lobe of the filter when represented in the frequency domain
  // Find total markers
  std::vector<std::vector<cv::Point> > contours;
  //cv::Mat binary = H_ > 0;
  cv::Mat markers = cv::Mat::zeros(H.size(), CV_8U);

  cv::findContours(cv::Mat(H > 0), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  auto contsBegin = contours.cbegin();
  for (auto conts = contsBegin, contsEnd = contours.cend(); conts != contsEnd; ++conts)
  {
    bool calcDistance(false);
    if(cv::pointPolygonTest(*conts, cv::Point(H.rows/2, H.cols/2), calcDistance) > 0)
    {
      cv::drawContours(markers, contours, std::distance(contsBegin, conts), cv::Scalar::all(255), -1);
      break;
    }
  }

  H.setTo(0, markers == 0);
  cv::blur(H, H_, cv::Size(9,9));

}
