/*
 * WaveletTransform.cpp
 *
 *  Created on: Oct 16, 2013
 *      Author: dailos
 */
#include "WaveletTransform.h"
#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"
#include "PDTools.h"

void udwd(const cv::Mat& imgOriginal, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes)
{
  wavelet_planes.clear();
  cv::Mat in;
  imgOriginal.copyTo(in);
  //discrete filter derived from scaling function. In our calculation a 1D spline of degree 3
  double m[] = {1.0, 4.0, 6.0, 4.0, 1.0};
  int row_elements = sizeof(m) / sizeof(m[0]);
  cv::Mat row_ref(row_elements, 1, cv::DataType<double>::type, m);
  double scale_factor(2.0);
  cv::Mat scaling_function = row_ref * row_ref.t();

  for(unsigned int nplane = 0; nplane < total_planes; ++nplane)
  {
    scaling_function = scaling_function / cv::sum(scaling_function).val[0];
    conv_flaw(in, scaling_function, residu);
    wavelet_planes.push_back(in - residu);
    //update variables
    residu.copyTo(in);
    cv::resize(scaling_function, scaling_function, cv::Size(0,0), scale_factor, scale_factor, cv::INTER_NEAREST);
  }
}

void swtSpectrums(const cv::Mat& imgSpectrums, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes)
{
  cv::Mat source;
  imgSpectrums.copyTo(source);
  
  wavelet_planes.clear();
  //discrete filter derived from scaling function. In our calculation a 1D spline of degree 3
  double m[] = {1.0, 4.0, 6.0, 4.0, 1.0};
  int row_elements = sizeof(m) / sizeof(m[0]);
  cv::Mat row_ref(row_elements, 1, cv::DataType<double>::type, m);
  double scale_factor(2.0);
  cv::Mat scaling_function = row_ref * row_ref.t();

  for(unsigned int nplane = 0; nplane < total_planes; ++nplane)
  {
    scaling_function = scaling_function / cv::sum(scaling_function).val[0];
    cv::Mat kernelPadded = cv::Mat::zeros(source.size(), source.depth());
    scaling_function.copyTo(selectCentralROI(kernelPadded, scaling_function.size()));

    cv::Mat kernelPadded_ft;
    cv::dft(kernelPadded, kernelPadded_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    cv::mulSpectrums(source, kernelPadded_ft.mul(kernelPadded.total()), residu, cv::DFT_COMPLEX_OUTPUT);

    wavelet_planes.push_back(source - residu);
    //update variables
    residu.copyTo(source);
    cv::resize(scaling_function, scaling_function, cv::Size(0,0), scale_factor, scale_factor, cv::INTER_NEAREST);
  }  
}

//stationary wavelet transform - a trous isotropic wavelet transform
void swt(const cv::Mat& imgOriginal, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes)
{
  wavelet_planes.clear();
  //discrete filter derived from scaling function. In our calculation a 1D spline of degree 3
  float m[] = { 1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16 };
  cv::Mat kernelLoG(5, 1, CV_32F, m);
  cv::Mat imgPrevious;
  imgOriginal.copyTo(imgPrevious);
  for(unsigned int nplane = 0; nplane < total_planes; ++nplane)
  {
    cv::Mat kernel = filterUpsampling(kernelLoG, nplane);
    cv::filter2D(imgPrevious, residu, -1, kernel * kernel.t());
    wavelet_planes.push_back(imgPrevious - residu);
    residu.copyTo(imgPrevious);
  }
}

cv::Mat filterUpsampling(const cv::Mat& src, const unsigned int &scale)
{
  cv::Mat dst = cv::Mat::zeros((src.rows*std::pow(2,scale))-(std::pow(2,scale)-1), (src.cols*std::pow(2,scale))-(std::pow(2,scale)-1), src.type());
  for(int i(0); i<src.rows; ++i)
  {
    for(int j(0); j<src.cols; ++j)
  	{
      dst.at<float>(i*std::pow(2,scale),j*std::pow(2,scale)) = src.at<float>(i,j);
    }
  }
  return dst;
}
