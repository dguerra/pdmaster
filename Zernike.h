/*
 * Zernike.h
 *
 *  Created on: Oct 3, 2013
 *      Author: dailos
 */

#ifndef ZERNIKES_H_
#define ZERNIKES_H_


#include <vector>
#include <string>
#include <complex>
#include <map>
#include <cmath>
#include <ctime>
#include "opencv2/opencv.hpp"

class Zernike
{
 public:

  
  Zernike(const double& r_c, const int& nph, const int& z_max = 20);
  /**
   * Default constructor of the Zernike class.
   */
  Zernike();

  /**
   * Returns a square phase map given by the sum of Zernike abberations
   * where the unit circle is maximized.
   * @param sideLength   The size in pixels of the output phase map
   */
  cv::Mat phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs);

  void analyse(const cv::Mat& sig, cv::Mat& z_coeffs);
  
  void synthesize(const cv::Mat& z_coeffs, cv::Mat& sig);
  
  void zernike_mn(const int& j,int &m,int &n);
  
  double zernike_covar(int i,int j);
  
  void polynomial(const double& r_c, const int& nph, const int& j, cv::Mat& z_j);
  
  //Circular pupil amplitude
  void circular_mask(const double& r_c, const int& nph, cv::Mat& c_mask);
  void circular_mask(cv::Mat& c_mask);
  /**
   * Returns a phase map given by a single Zernike polynomial
   *
   * @param j           The Zernike single index
   * @param sideLength   The size in pixels of the output phase map
   */
  cv::Mat phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength, const bool& unit_rms = true, const bool& c_mask = true);
  
  const std::vector<cv::Mat>& base() const {return base_;};

 private:
  double radious_px_;
  int side_px_;
  std::vector<cv::Mat> base_;
};

#endif /* ZERNIKES_H_ */
