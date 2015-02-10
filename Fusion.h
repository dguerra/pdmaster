/*
 * Fusion.h
 *
 *  Created on: Jan 16, 2015
 *      Author: dailos
 */

#ifndef FUSION_H_
#define FUSION_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

void fuse(const cv::Mat& A, const cv::Mat& B, const double& sigmaNoise, cv::Mat& fusedImg);
void swtSpectrums_(const cv::Mat& imgSpectrums, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes = 4);
void probabilisticMask_(const cv::Mat& data, const cv::Mat& noise, const cv::Size& windowSize, cv::Mat& mask);
void waveletsNoiseSimulator(const double& sigmaNoise, const unsigned int& total_planes, const cv::Size& simulationSize, std::vector<double>& wNoiseFactor);

#endif /* FUSION_H_ */