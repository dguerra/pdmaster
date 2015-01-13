/*
 * GetStep.h
 *
 *  Created on: Oct 30, 2014
 *      Author: dailos
 */

#ifndef GETSTEP_H_
#define GETSTEP_H_
#include "opencv2/opencv.hpp"

int getstep(cv::Mat& c, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& diversityPhase, const cv::Mat& pupilAmplitude,
		const unsigned int& pupilSideLength, const cv::Mat& zernikesInUse, const cv::Mat& alignmentSetup,
		const std::map<unsigned int, cv::Mat>& zernikeCatalog,
		const double& pupilRadiousP, const std::vector<double>& meanPowerNoise, double& lmPrevious,
		unsigned int& numberOfNonSingularities, double& singularityThresholdOverMaximum, cv::Mat& dc);

#endif /* GETSTEP_H_ */
