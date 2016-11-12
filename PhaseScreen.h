/*
 * PhaseScreen.h
 *
 *  Created on: Jan 27, 2014
 *      Author: dailos
 */

#ifndef PHASESCREEN_H_
#define PHASESCREEN_H_

#include <iostream>
#include "opencv2/opencv.hpp"

class PhaseScreen
{
public:
  PhaseScreen();
  virtual ~PhaseScreen();
  cv::Mat fractalMethod(const unsigned int& nw, const double& r0, const double& L0);
  
private:

};


#endif /* PHASESCREEN_H_ */
