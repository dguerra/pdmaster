/*
 * MonitorLog.h
 *
 *  Created on: Nov 18, 2013
 *      Author: dailos
 */

#ifndef MONITORLOG_H_
#define MONITORLOG_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

class Record
{
public:
  Record();
  virtual ~Record();
  void printValues();
  Record(unsigned int iteration, double lmCurrent, double lmIncrement, cv::Mat c, double cRMS, double dcRMS,
         unsigned int numberOfNonSingularities, double singularityThresholdOverMaximum) : //Initialization list
           iterationNumber_(iteration), lm_(lmCurrent), lmIncrement_(lmIncrement), c_(c), cRMS_(cRMS),
           dcRMS_(dcRMS), numberOfNonSingularities_(numberOfNonSingularities), singularityThresholdOverMaximum_(singularityThresholdOverMaximum){};
private:
  unsigned int iterationNumber_;
  double lm_;
  double lmIncrement_;
  cv::Mat c_;
  double cRMS_;
  double dcRMS_;
  unsigned int numberOfNonSingularities_;
  double singularityThresholdOverMaximum_;
};

class MonitorLog
{
public:
  MonitorLog();
  virtual ~MonitorLog();
  void add(Record r){history_.push_back(r);};
  void printValues();
private:
  std::vector<Record> history_;
};

#endif /* MONITORLOG_H_ */

