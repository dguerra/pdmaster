/*
 * Zernikes.h
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

class Zernikes
{
 public:

  /**
   * Default constructor of the Zernikes class.
   */
  Zernikes();

  /**
   * Creates a new Zernikes object by the coeficient vector.
   *
   * @param ZernikeCoefs        Vector of Zernike coefficients: rms at the
   *                            wavefront over the unit circular pupil [m@WF].
   *                            In the case of hexagonal pupils, the
   *                            coefficients are still defined as rms over the
   *                            unit circular pupil, where the hexagonal
   *                            pupil is inscribed inside the unit circle.
   */
  Zernikes(const std::map<unsigned int, double>& zernikeCoefs);


  /**
   *    Returns the wavefront abberation of a given Zernike polynom
   *    at a given point.
   *
   * @param j      The Zernike single index
   * @param rho    The radius of the points polar coordinates. The radius has
   *               to be normalized to the circle radius (For the hexagonal
   *               segments case, this is 1/sqrt(3) the segment center distance)
   *               (IN).
   * @param theta  The angle of the points polar coordinates (IN).
   *               The convention used here and adopted by the OSA has
   *               x "horizontal", y "vertical", and theta is measured
   *               counter-clockwise from x-axis (i.e. right-handed coordinate
   *               system). More traditional notation measures theta clockwise
   *               from y-axis.
   */
  static double pointZernike(const unsigned int& j, const double& rho, const double& theta);


  /**
   * Returns a square phase map given by the sum of Zernike abberations
   * where the unit circle is maximized.
   * @param sideLength   The size in pixels of the output phase map
   */
  static cv::Mat phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs);

  static cv::Mat zernike_function(const double& lambda, const double& r_c, const int& nph, const int& j, const double& angle);
  
  static cv::Mat pupilAmplitude(const double& r_c, const int& nph);
  /**
   * Returns a phase map given by a single Zernike polynomial
   *
   * @param j           The Zernike single index
   * @param sideLength   The size in pixels of the output phase map
   */
  static cv::Mat phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength);


  static std::vector<cv::Mat> zernikeBase(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength);
  static std::map<unsigned int, cv::Mat> buildCatalog(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength);

  /*
   * Calculates n and m from the single index j.
   *
   * @param j   The single index (IN)
   * @param n   Radial order (OUT)
   * @param m   Azimuthal frequency m (OUT)
   */
  static void getNM(const unsigned int& j, unsigned int& n, int& m);

  //std::vector<cv::Mat> catalog();

  std::map<unsigned int, double> zernikeCoefs()const {return zernikeCoefs_;};
  void setCoef(const unsigned int& zernikeIndex, const double& zernikeCoef);
  double getCoef(const unsigned int& zernikeIndex);
 private:
  /*
   * The Zernike coeficient vector
   */
  std::map<unsigned int, double> zernikeCoefs_;

};

#endif /* ZERNIKES_H_ */
