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
void test_IHT();
void iterativeHardThresholding( const cv::Mat& observation, const cv::Mat& measurement, cv::Mat&  x0, const unsigned int& sparsity, 
                                const double& mu, const unsigned int& numberOfIterations);
                                
#endif /* COMPRESSEDSENSING_H_ */
