/*
 * CompressedSensing.h
 *
 *  Created on: Sep 01, 2015
 *      Author: dailos
 */

#ifndef COMPRESSEDSENSING_H_
#define COMPRESSEDSENSING_H_

#include <iostream>
#include "opencv2/opencv.hpp"

void dctMatrix(const unsigned int& m, const unsigned int& n, cv::Mat& mat);
void matchingPursuit();
void iterativeHardThresholding();
#endif /* COMPRESSEDSENSING_H_ */
