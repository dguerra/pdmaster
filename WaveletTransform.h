/*
 * WaveletTransform.h
 *
 *  Created on: Oct 16, 2013
 *      Author: dailos
 */

#ifndef WAVELETTRANSFORM_H_
#define WAVELETTRANSFORM_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

//stationary wavelet transform
void swt(const cv::Mat& imgOriginal, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes = 4);
void udwd(const cv::Mat& imgOriginal, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes = 4);
void swtSpectrums(const cv::Mat& imgSpectrums, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes = 4);
cv::Mat filterUpsampling(const cv::Mat& src, const unsigned int &scale);

#endif /* WAVELETTRANSFORM_H_ */
