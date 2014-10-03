/*
 * ImageQualityMetric.h
 *
 *  Created on: Apr 30, 2014
 *      Author: dailos
 */

#ifndef IMAGEQUALITYMETRIC_H_
#define IMAGEQUALITYMETRIC_H_

#include "opencv2/opencv.hpp"

class ImageQualityMetric
{
public:
  ImageQualityMetric();
  virtual ~ImageQualityMetric();
  //Structural Similarity Index Metric
  double correlationCoefficient(const cv::Mat& x, const cv::Mat& y);
  double meanSquareError(const cv::Mat& x, const cv::Mat& y);
  double covariance(const cv::Mat& x, const cv::Mat& y);
  double ssim(const cv::Mat& x, const cv::Mat& y);
  cv::Scalar mssim( const cv::Mat& i1, const cv::Mat& i2);
private:

};
#endif /* IMAGEQUALITYMETRIC_H_ */

