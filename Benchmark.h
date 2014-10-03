/*
 * Benchmark.h
 *
 *  Created on: Jun 3, 2014
 *      Author: dailos
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "opencv2/opencv.hpp"

void process_AWMLE();
void createModifiedHanningWindow(cv::Mat& modifiedHanningWindow, const int& sideLength, const double& apodizedAreaPercent, int datatype);



#endif /* BENCHMARK_H_ */
