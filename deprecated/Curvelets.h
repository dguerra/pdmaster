/*
 * Curvelets.h
 *
 *  Created on: March 30, 2015
 *      Author: dailos
 */

#ifndef CURVELETS_H_
#define CURVELETS_H_

#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "curvelab/fdct_wrapping.hpp"
#include "curvelab/fdct_wrapping_inline.hpp"

class Curvelets
{
  public:
    Curvelets();
    virtual ~Curvelets();
    static void fdct(const cv::Mat& I, std::vector< vector<cv::Mat> >& c,  const bool& real_coeffs = true);
    static void ifdct(std::vector< vector<cv::Mat> >& c, cv::Mat& complexI, int m, int n, const bool& real_coeffs = true);
    static double l1_norm(const std::vector< vector<cv::Mat> >& c);
  private:
    static void r2c(std::vector< std::vector<cv::Mat> >& c);
    static void c2r(std::vector< std::vector<cv::Mat> >& c);
    std::vector< vector<cv::Mat> > c;
};

#endif /* CURVELETS_H_ */
