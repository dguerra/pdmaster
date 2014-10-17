/*
 * AWMLE.h
 *
 *  Created on: Apr 15, 2014
 *      Author: dailos
 */

#ifndef AWMLE_H_
#define AWMLE_H_
#include "opencv2/opencv.hpp"

void AWMLE(const cv::Mat& img, const cv::Mat& psf, cv::Mat& object, const double& sigmaNoise, const unsigned int& nplanes = 4);
void perElementFiltering(const cv::Mat& img, const cv::Mat& projection, cv::Mat& out, const double& sigmaNoise);
void calculpprima(const cv::Mat& img, const cv::Mat& prj, const double& sigmaNoise, cv::Mat& modifiedimage, double& likelihood);
void waveletNoise(const double& sigma, const unsigned int& nplanes, const cv::Size& simulationSize, std::vector<double>& wNoiseFactor);
void probabilisticMask(const cv::Mat& data, const cv::Mat& noise, const cv::Size& windowSize, cv::Mat& mask);
void getLocalStdDev(const cv::Mat& i, const cv::Size& windowSize, cv::Mat& sigma2);

#endif /* AWMLE_H_ */
