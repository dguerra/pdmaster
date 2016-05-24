/*
 * Optics.h
 *
 *  Created on: Jan 31, 2014
 *      Author: dailos
 */

#ifndef OPTICS_H_
#define OPTICS_H_
#include "opencv2/opencv.hpp"
//Rename module as simply "Optics"

class Optics
{
public:
  Optics(const cv::Mat& phase, const cv::Mat& amplitude);
  virtual ~Optics();
  cv::Mat otf()const {return otf_;};
  cv::Mat generalizedPupilFunction()const;
  //cv::Mat otfNormalizationFactor()const {return otfNormalizationFactor_;};
private:
  Optics();  //private default contructor, only parameters contructor is allowed
  void compute_OTF_(const cv::Mat& phase, const cv::Mat& amplitude, cv::Mat& otf);
  void compute_GeneralizedPupilFunction_(const cv::Mat& phase, const cv::Mat& amplitude, cv::Mat& generalizedPupilFunction);
  cv::Mat otf_;
  cv::Mat generalizedPupilFunction_;
  cv::Mat otfNormalizationFactor_;
  double pupilRadious_;
};

#endif /* OPTICS_H_ */
