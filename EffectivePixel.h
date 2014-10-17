/*
 * EffectivePixel.h
 *
 *  Created on: Jan 30, 2014
 *      Author: dailos
 */

#ifndef EFFECTIVEPIXEL_H_
#define EFFECTIVEPIXEL_H_

#include <vector>
#include <string>
#include <complex>
#include <iostream>

#include "CustomException.h"
//Alternative name: "ActualPixel", "TruePixel" to refer to the idea of being a more precise pixel value
//See Effective pixel drawing for more info
class EffectivePixel
{
public:
  EffectivePixel();
  EffectivePixel(const std::complex<double>& pixelCoord, const double& pixelSize, const double& radiousOfTheCircle = 1.0);
  virtual ~EffectivePixel();

  //getters
  double effectiveArea(){return effectiveArea_;};
  std::complex<double> effectiveCoord(){return effectiveCoord_;};

  /*
   * When pixel lies in the edge of the circle,
   * we take the center of mass of the portion of pixel inside as the coordenate
   * and the value is proportional to the area inside the circle
   *
   * */
  void fromActualPixel(const std::complex<double>& pixelCoord, const double& pixelSize);

private:
 /*
  * Gives the area of a polygon.
  * The vector contains every point of the polygon
  * The order in wich the algorithm takes the points is important
  * */
  double polygonArea_() const;

 /*
  * Calculates the center of mass of a polygon
  * The vector contains every point of the polygon
  * The order in wich the algorithm takes the points is important
  * */
  std::complex<double> polygonCenterOfMass_() const;

 /*
  * Calculates intersection points between a segment and a circle cetered at the origin and radious one
  * and adds them to polygonContour vector
  * */
  void findIntersectionPoints_(const std::complex<double>& pointA, const std::complex<double>& pointB);

 /*
  * Center of mass of portion of the actual pixel that lies within the boundaries zone
  * */
  std::complex<double> effectiveCoord_;

 /*
  * Area of the actual pixel that lies within boundaries zone
  * */
  double effectiveArea_;

 /*
  * Center of the circle that defines the boundaries to consider
  * */
  std::complex<double> centerOfTheCircle_;

 /*
  * Radious of the circle that defines the boundaries to consider
  * */
  double radiousOfTheCircle_;

 /*
  * Points that define the polygon which corresponds to the piece of pixel that lies within the boundaries
  * */
  std::vector<std::complex<double> > polygonContour_;
};

#endif /* EFFECTIVEPIXEL_H_ */
