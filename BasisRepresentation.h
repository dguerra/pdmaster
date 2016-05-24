/*
 * BasisRepresentation.h
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

class BasisRepresentation
{
 public:

  /**
   * Default constructor of the BasisRepresentation class.
   */
  BasisRepresentation();

  /**
   * Creates a new BasisRepresentation object by the coeficient vector.
   *
   * @param ZernikeCoefs        Vector of Zernike coefficients: rms at the
   *                            wavefront over the unit circular pupil [m@WF].
   *                            In the case of hexagonal pupils, the
   *                            coefficients are still defined as rms over the
   *                            unit circular pupil, where the hexagonal
   *                            pupil is inscribed inside the unit circle.
   */
  BasisRepresentation(const std::map<unsigned int, double>& zernikeCoefs);

  /**
   * Returns a square phase map given by the sum of Zernike abberations
   * where the unit circle is maximized.
   * @param sideLength   The size in pixels of the output phase map
   */
  static cv::Mat phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs);

  static cv::Mat zernike_analysis(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& phase_signal);
  
  static void zernike_mn(const int& j,int &m,int &n);
  
  static double zernike_covar(int i,int j);
  
  static cv::Mat zernike_function(const double& r_c, const int& nph, const int& j);
  
  //Circular pupil amplitude
  static cv::Mat circular_mask(const double& r_c, const int& nph);
  /**
   * Returns a phase map given by a single Zernike polynomial
   *
   * @param j           The Zernike single index
   * @param sideLength   The size in pixels of the output phase map
   */
  static cv::Mat phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength, const bool& unit_rms = true, const bool& c_mask = true);

  /*
   * Calculates n and m from the single index j.
   *
   * @param j   The single index (IN)
   * @param n   Radial order (OUT)
   * @param m   Azimuthal frequency m (OUT)
   */
  static void getNM(const unsigned int& j, unsigned int& n, int& m);

 private:
  /*
   * The Zernike coeficient vector
   */
  std::map<unsigned int, double> zernikeCoefs_;

};

#endif /* ZERNIKES_H_ */
