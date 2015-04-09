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
    void fdct_wrapping(const cv::Mat& complexI, std::vector< vector<cv::Mat> >& c);
    void ifdct_wrapping(std::vector< vector<cv::Mat> >& c, cv::Mat& complexI);
    void fdct_wrapping_r2c(std::vector< std::vector<cv::Mat> >& c);
    void fdct_wrapping_c2r(std::vector< std::vector<cv::Mat> >& c);
    double l1_norm(const std::vector< vector<cv::Mat> >& c);
  private:
    std::vector< vector<cv::Mat> > c;
};

#endif /* CURVELETS_H_ */
