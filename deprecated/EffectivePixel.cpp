/*
 * EffectivePixel.cpp
 *
 *  Created on: Jan 30, 2014
 *      Author: dailos
 */

#include "EffectivePixel.h"
#include <algorithm>

EffectivePixel::EffectivePixel()
{
  effectiveArea_ = 0.0;
  centerOfTheCircle_ = std::complex<double>(0.0,0.0);
  radiousOfTheCircle_ = 1.0;
}

EffectivePixel::EffectivePixel(const std::complex<double>& pixelCoord, const double& pixelSize, const double& radiousOfTheCircle)
{
  //WE only consider pixelSize = 1 and circle radious expressed in pixels units
  centerOfTheCircle_ = std::complex<double>(0.0,0.0);
  radiousOfTheCircle_ = radiousOfTheCircle;
  fromActualPixel(pixelCoord, pixelSize);
}

EffectivePixel::~EffectivePixel()
{
  // TODO Auto-generated destructor stub
}

void EffectivePixel::fromActualPixel(const std::complex<double>& pixelCoord, const double& pixelSize)
{
  //The pixel center lies within the edgeThickness of the circle
  //Analyze further: how many and which pixels corners lie within the circle frontier
  std::complex<double> upperLeft  = pixelCoord + std::complex<double>(-pixelSize/2,pixelSize/2);
  std::complex<double> upperRight = pixelCoord + std::complex<double>(pixelSize/2,pixelSize/2);
  std::complex<double> lowerRight = pixelCoord + std::complex<double>(pixelSize/2,-pixelSize/2);
  std::complex<double> lowerLeft  = pixelCoord + std::complex<double>(-pixelSize/2,-pixelSize/2);
  //note that the order in which the points are added to the polygon vector is important
  polygonContour_.clear();  //start up defining the polygon wich defines the part of the pixel inside the cicle
  if(std::abs(upperLeft)<=radiousOfTheCircle_) polygonContour_.push_back(upperLeft);
  findIntersectionPoints_(upperLeft, upperRight);

  if(std::abs(upperRight)<=radiousOfTheCircle_) polygonContour_.push_back(upperRight);
  findIntersectionPoints_(upperRight, lowerRight);

  if(std::abs(lowerRight)<=radiousOfTheCircle_) polygonContour_.push_back(lowerRight);
  findIntersectionPoints_(lowerRight, lowerLeft);

  if(std::abs(lowerLeft)<=radiousOfTheCircle_) polygonContour_.push_back(lowerLeft);
  findIntersectionPoints_(lowerLeft, upperLeft);

  if(!polygonContour_.empty())
  {
    effectiveArea_ = std::abs(polygonArea_())/(pixelSize*pixelSize);
    effectiveCoord_ = polygonCenterOfMass_();
  }
  else if(polygonContour_.empty() && std::abs(pixelCoord)<=radiousOfTheCircle_)
  {
    effectiveArea_ = 3.1415;  //The whole circle lies within the pixel (rare case)
    effectiveCoord_ = pixelCoord;
  }
  else if(polygonContour_.empty() && std::abs(pixelCoord)>=radiousOfTheCircle_)
  {  //not exaclty that, if pixel is greater than circle, pixel coordintates could be outside the circle
    //and overlaps at the same time
    effectiveArea_ = 0.0;
    effectiveCoord_ = pixelCoord;
  }
}

double EffectivePixel::polygonArea_() const
{
  double area(0.0);
  auto polyAlgorithm = [&area](std::complex<double> b, std::complex<double> a)->std::complex<double>
  {
    area += a.real()*b.imag()-a.imag()*b.real();
    return a;
  };

  if(!polygonContour_.empty())
  {
    std::accumulate(polygonContour_.begin(), polygonContour_.end(), polygonContour_.back(), polyAlgorithm);
  }
  //beware that the result is with sign! In some cases the sign of the result can be useful.
  return area/2;
}

std::complex<double> EffectivePixel::polygonCenterOfMass_() const
{
  double xPosition(0.0), yPosition(0.0);
  double area = polygonArea_();
  auto centerOfMassAlgorithm = [&xPosition, &yPosition](std::complex<double> b, std::complex<double> a)->std::complex<double>
  {
    xPosition += (a.real()+b.real())*(a.real()*b.imag()-b.real()*a.imag());
    yPosition += (a.imag()+b.imag())*(a.real()*b.imag()-b.real()*a.imag());
    return a;
  };

  if(!polygonContour_.empty())
  {
    std::accumulate(polygonContour_.begin(), polygonContour_.end(), polygonContour_.back(), centerOfMassAlgorithm);
  }

  return std::complex<double>(xPosition/(6*area), yPosition/(6*area));
}

void EffectivePixel::findIntersectionPoints_(const std::complex<double>& pointA, const std::complex<double>& pointB)
{
  //Calculates the intersection point between a segment A-B and a circumference
  double rotAngle(0.0);  //rotate the scene
  if(pointA.real()-pointB.real() == 0)
  {  //if segment is vertical, rotate the scene to avoid indetermination due to infinite slope
      rotAngle = 1.3;  //some trivial amount of rads
  }
  //rotate the scene if needed
  std::complex<double> segmentBegin = pointA * std::exp(std::complex<double>(0, rotAngle));
  std::complex<double> segmentEnd = pointB * std::exp(std::complex<double>(0, rotAngle));
  std::complex<double> center = centerOfTheCircle_ * std::exp(std::complex<double>(0, rotAngle));


  double m = (segmentEnd.imag()-segmentBegin.imag())/(segmentEnd.real()-segmentBegin.real());
  double c = segmentBegin.imag()-(m*segmentBegin.real());

  double A = std::pow(m,2) + 1;
  double B = 2*(m*c-m*center.imag()-center.real());
  double C = std::pow(center.imag(),2) - std::pow(radiousOfTheCircle_,2) + std::pow(center.real(),2) - 2*c*center.imag() + std::pow(c,2);

  double segmentSize(std::abs(segmentBegin-segmentEnd));
  double discriminant = std::pow(B,2)-(A*C*4);

  if(discriminant < 0)
  {
    //no intersection points, adds nothing
  }
  else if(discriminant == 0)
  {
    //only one intersection point
    std::complex<double> intersectionPoint( (-B + std::sqrt(discriminant))/(2*A), m*((-B + std::sqrt(discriminant))/(2*A))+c);
    if(std::abs(intersectionPoint-segmentBegin)<=segmentSize && std::abs(intersectionPoint-segmentEnd)<=segmentSize)
    {
      //the point belong to the segment, rotate back first
      polygonContour_.push_back(intersectionPoint * std::exp(std::complex<double>(0, -rotAngle)));
    }
  }
  else if(discriminant > 0)
  {
    //two intersection points
    std::complex<double> intersectionPointA((-B + std::sqrt(discriminant))/(2*A), m*((-B + std::sqrt(discriminant))/(2*A))+c);
    if(std::abs(intersectionPointA-segmentBegin)<=segmentSize && std::abs(intersectionPointA-segmentEnd)<=segmentSize)
    {  //rotate back first
      polygonContour_.push_back(intersectionPointA * std::exp(std::complex<double>(0, -rotAngle)));
    }
    std::complex<double> intersectionPointB((-B - std::sqrt(discriminant))/(2*A), m*((-B - std::sqrt(discriminant))/(2*A))+c);
    if(std::abs(intersectionPointB-segmentBegin)<=segmentSize && std::abs(intersectionPointB-segmentEnd)<=segmentSize)
    {  //rotate back first
      polygonContour_.push_back(intersectionPointB * std::exp(std::complex<double>(0, -rotAngle)));
    }
  }
}
