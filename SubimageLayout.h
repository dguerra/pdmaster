/*
 * SubimageLayout.h
 *
 *  Created on: Jan 27, 2014
 *      Author: dailos
 */

#ifndef SUBIMAGELAYOUT_H_
#define SUBIMAGELAYOUT_H_

#include <iostream>
#include "opencv2/opencv.hpp"
//other names Subimage or SubimageIterator

class SubimageLayout
{
public:
  SubimageLayout();
  virtual ~SubimageLayout();
  void navigateThrough();
  void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow, const int& sideLength, const double& apodizedAreaPercent, int datatype);
//  bool subimageQueueIsEmpty(){return subimageQueue_.empty();};
//  cv::Mat nextSubimage(){return subimageQueue_.pop_front();};

private:
  int subimageSize_;  //size of the box to be restored
  int subimageStepXSize_;
  int subimageStepYSize_;
  double apodizationPercent_;
  void populateSubimageQueue_();
  std::vector<cv::Mat> subimageQueue_;

};

class Subimage
{
public:
  Subimage();
  virtual ~Subimage();

private:
  std::pair<unsigned long, unsigned long> subimageLocation_;
};

#endif /* SUBIMAGELAYOUT_H_ */
