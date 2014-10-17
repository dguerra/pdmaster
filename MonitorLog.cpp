/*
 * Monitor.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: dailos
 */

//Other names MagnitudeLog, recorder, etc
//This class will record and store information for every loop pass, to study convergence
//It will also plot graphs with their values

#include "MonitorLog.h"
//other names benchmark
MonitorLog::MonitorLog()
{
  // TODO Auto-generated constructor stub

}

MonitorLog::~MonitorLog()
{
  // TODO Auto-generated destructor stub
}

Record::Record()
{
  iterationNumber_ = 0;
  lm_ = 0.0;
  lmIncrement_ = 0.0;
  cRMS_ = 0.0;
  dcRMS_ = 0.0;
  numberOfNonSingularities_ = 0;
  singularityThresholdOverMaximum_ = 0.0;
}

Record::~Record()
{
  // TODO Auto-generated destructor stub
}

void Record::printValues()
{
  std::cout << "iteration: " << iterationNumber_ << std::endl;
  std::cout << "lm_: " << lm_ << std::endl;
  std::cout << "lmIncrement_: " << lmIncrement_ << std::endl;
  std::cout << "c_: " << c_ << std::endl;
  std::cout << "cRMS_: " << cRMS_ << std::endl;
  std::cout << "dcRMS_: " << dcRMS_ << std::endl;
  std::cout << "numberOfNonSingularities_: " << numberOfNonSingularities_ << std::endl;
  std::cout << "singularityThresholdOverMaximum_: " << singularityThresholdOverMaximum_ << std::endl;
  std::cout << std::endl;
}

void MonitorLog::printValues()
{
  //for_each(history_.begin(), history_.end(), printValues());
}
