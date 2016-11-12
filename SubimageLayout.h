/*
 * SubimageLayout.h
 *
 *  Created on: Jan 27, 2014
 *      Author: dailos
 */

#ifndef SUBIMAGELAYOUT_H_
#define SUBIMAGELAYOUT_H_

#include <iostream>
#include <memory>
#include <utility>
#include "opencv2/opencv.hpp"
#include "Metric.h"
//other names Subimage or SubimageIterator

class SubimageLayout
{
public:
  SubimageLayout();
  virtual ~SubimageLayout();
  // navigateThrough
  void fromPhaseScreen();
  void dataSimulator();
  void aberrate(const cv::Mat& img, const cv::Mat& aberrationPhase, const double& pupilRadious, const double& sigmaNoise, cv::Mat& aberratedImage);
  cv::Mat atmospheric_zernike_coeffs(const unsigned int& z_max, const double& D, const double& r0);
  void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow, const int& sideLength, const double& apodizedAreaPercent, int datatype);
  void computerGeneratedImage();
  //std::vector<std::tuple<std::pair<cv::Range,cv::Range>, cv::Mat, std::unique_ptr<Metric> > > tileStack_;
  std::vector<std::tuple<std::pair<cv::Range,cv::Range>, cv::Mat, Metric> > tileStack_;
  
private:
  cv::Mat canvas_;   //blank canvas to set locations and sizes of tiles

};


#endif /* SUBIMAGELAYOUT_H_ */
