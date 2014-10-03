/*
 * FITS.h
 *
 *  Created on: Feb 28, 2014
 *      Author: dailos
 */

#ifndef FITS_H_
#define FITS_H_

#include <iostream>
#include "opencv2/opencv.hpp"


void readFITS(const std::string& fitsname, cv::Mat& image);
void writeFITS(const cv::Mat& cvImage, const std::string& filename);

#endif /* FITS_H_ */
