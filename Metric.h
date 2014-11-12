/*
 * NoiseFilter.h
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */

#ifndef METRIC_H_
#define METRIC_H_

#include <iostream>
#include "OpticalSystem.h"
#include "opencv2/opencv.hpp"

class Metric
{
public:
  Metric();
  virtual ~Metric();
  
private:
    void computeGrandient_(const std::vector<OpticalSystem>& OS, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase);
};

#endif /* METRIC_H_ */

