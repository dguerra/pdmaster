//============================================================================
// Name        : Zernikes.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Zernikes.h"
#include "EffectivePixel.h"
#include "PDTools.h"
#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"
#include "CustomException.h"

using namespace std;
//constexpr double PI_4 = -acos(0.0)/2;

//The default constructor of the Zernikes class
template<typename T>
Zernikes<T>::Zernikes()
{
//  setNameMap_();
}

//Constructor with a vector of Zernike coeficients.
template<typename T>
Zernikes<T>::Zernikes
(const std::map<unsigned int, double>& zernikeCoefs)
{
  zernikeCoefs_ = zernikeCoefs;
//  setNameMap_();
}

//Returns the wavefront abberation of a given Zernike polynom at a given point.
//Default version with input (j, rho, theta)
template<typename T>
T Zernikes<T>::pointZernike(const unsigned int& j, const double& rho, const double& theta)
{
  if(rho < 0) throw CustomException("Rho cannot be negative");
  if(j == 0) throw CustomException("Zernike index cannot be zero");

  try
  {
    double Znm = 0;
    if (rho <= 1)//for rho>1, Znm=0
    {
        //From j, calculate n and m
        unsigned int n;
        int m;
        getNM(j, n, m);
        //std::cout << "n,m: " << n << "," << m << std::endl;
        //Calculate Znm (the Zernike for n and m) including the corresponding
        //  normalization.
        double Nnm = std::sqrt(2.0*(n+1.0)/(1.0+(m==0))); //Normalization Factor
        //std::cout << "j = 8; n = " + std::to_string(n) + "; m = " + std::to_string(m) << std::endl;

        //std::cout << "Nnm: " << Nnm << std::endl;
        //Get the radial term
        for (int s=0; s <= 0.5*(n-std::abs(m)); ++s)
        {
          Znm += std::pow(-1.0,s) * factorial(n-s) * std::pow(rho,(n-2*s))
                        /( factorial(s)
                          *factorial(static_cast<int>(0.5*(n+std::abs(m))-s))
                          *factorial(static_cast<int>(0.5*(n-std::abs(m))-s)) );
        }
        //Multiply with the expansion coefficioent (*zernIt),
        //         the normalization factor, and the sinosoidal term
        Znm*=Nnm*((m>=0)*cos(m*theta)-(m<0)*sin(m*theta));   //GTC version

    }
    return Znm;
  }
  catch (std::exception &e)
  {
    throw;
  }
}

template<typename T>
cv::Mat Zernikes<T>::phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs)
{
  if(coeffs.cols == 1 && coeffs.type() == cv::DataType<double>::type)
  {
    cv::Mat thePhaseMapSum = cv::Mat::zeros(sideLength, sideLength, cv::DataType<T>::type);

    for(auto cIt = coeffs.begin<double>(), cEnd = coeffs.end<double>(); cIt != cEnd; ++cIt)
    {
      if((*cIt) != 0.0)
      {
        //cv::accumulate((*cIt) * phaseMapZernike(std::distance(coeffs.begin<double>(), cIt) + 1, sideLength, radiousLength), thePhaseMapSum);
        thePhaseMapSum += (*cIt) * phaseMapZernike(std::distance(coeffs.begin<double>(), cIt) + 1, sideLength, radiousLength);
      }
    }

    return thePhaseMapSum;
  }
  else
  {
    throw CustomException("Zernikes: coeffs is single column vector of doubles.");
  }
}

template<typename T>
cv::Mat Zernikes<T>::phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength)
{
  if (j == 0) throw CustomException("Zernikes: j must not be zero");
  //if(radiousLength>sideLength/2) throw CustomException("Zernikes: radious longer that half the side");
  unsigned long nPixels(0);
  cv::Mat thePhaseMap = cv::Mat::zeros(sideLength, sideLength, cv::DataType<T>::type);
  //Loop over all pixels in thePhaseMap. We need to know the coordinates.
  //An explanation of the coordinate conventions used:
  //A 1dim array with dim=4 has pixels with carth.center coordinates 0,1,2,3.
  //The carth. coordinate of the center is at 1.5.
  //The maximum radius, normalized to 1 for a 1dim array is
  //at carth. coord -0.5 and +3.5.

  double center = (sideLength-1.)/2.;

  //Half diagonal of the pixel square
  double edgeThickness = std::sqrt(1/2);

  for (unsigned int xIt = 0; xIt < sideLength; ++xIt)
  {
    for (unsigned int yIt = 0; yIt < sideLength; ++yIt)
    {
      double xCoord = xIt - center;
      double yCoord = yIt - center;
      std::complex<double> pixelCoord(xCoord, yCoord);

      //std::complex<double> theCoord(pixelCoord);
      double pixelArea(1.0);

      //first in case the pixel is out side
      if(std::abs(pixelCoord) <= (radiousLength + edgeThickness))
      {
        if(std::abs(pixelCoord)>=(radiousLength - edgeThickness))
        {//if the pixel lies in the thickness of the circle we take
          //the center of mass of the part of the pixel inside the cicle as the coordinates
          //and its value is proportial to the area inside the circle
          //We only consider pixelSize = 1.0 and radiousLength expressed in pixel units
          EffectivePixel ePixel(pixelCoord, 1.0, radiousLength);
          pixelArea = ePixel.effectiveArea();
          pixelCoord = ePixel.effectiveCoord();
        }

        //else if(std::abs(pixelCoord)<(radiousLength - edgeThickness)) thePhaseMap.at<T>(xIt,yIt) = something;
        thePhaseMap.at<T>(xIt,yIt) = pixelArea * pointZernike(j, std::abs(pixelCoord)/radiousLength, std::arg(pixelCoord));
        nPixels++;
      }
    }
  }

  //Not suere if it is needed to do it
  double rmsZ = std::sqrt(cv::sum(thePhaseMap.mul(thePhaseMap)).val[0]/nPixels);
  return thePhaseMap/rmsZ;
}

template<typename T>
std::map<unsigned int, cv::Mat> Zernikes<T>::buildCatalog(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength)
{
  //if(radiousLength <= sideLength/2)
  {
    std::map<unsigned int, cv::Mat> catalog;
    for(unsigned int currentIndex=1; currentIndex <= maximumZernikeIndex; ++currentIndex)
    {
      catalog.insert(std::make_pair(currentIndex,phaseMapZernike(currentIndex, sideLength, radiousLength)));
    }

    return catalog;
  }
  //else
  {
    //throw CustomException("Zernikes: radiousLength larger than half the side of the image.");
  }
}


//Same as buildCatalog, but this is a std::vector of zernike polonomials instead of a std::map
template<typename T>
std::vector<cv::Mat> Zernikes<T>::zernikeBase(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength)
{
  std::vector<cv::Mat> base;
  for(unsigned int currentIndex=1; currentIndex <= maximumZernikeIndex; ++currentIndex)
  {
    base.push_back(phaseMapZernike(currentIndex, sideLength, radiousLength));
  }

  return base;
}


///Noll ordering scheme
template<typename T>
void Zernikes<T>::getNM(const unsigned int& j, unsigned int& n, int& m)
{
  if (j == 0)
    throw CustomException("j must not be zero");
  n = static_cast<unsigned int>(std::ceil((-3.+sqrt(9.+8.*(j-1)))/2.));
  //std::cout << "n: " << n << std::endl;
  //Calculate the number of modes up to this n (excluding modes in this n).
  //The formula is smallerModes = Sum_n(n). This can be derived from the
  //number of modes with a given n, which is n+1.
  unsigned int smallerModes = 0;
  for (unsigned int i = 1; i <= n; ++i)
  {
    smallerModes += i;
  }

  //std::cout << "smallerModes: " << smallerModes << std::endl;
  //Find the position i within this n (i>=1)
  int i = j - smallerModes;
  //trace_.out("getNM: j=%i. n=%i, smallerModes=%i, i=%i",j,n,smallerModes,i);
  //Find the absolute value and sign of m
  if (n%2 == 0) //n is even
  {
    if (i%2 == 0) //i is even
    {
      m = i;
    }
    else
    {
      m = i - 1;
    }
  }
  else        //n is odd
  {
    if (i%2 == 0) //i is even
    {
      m = i - 1;
    }
    else
    {
      m = i;
    }
  }

//  if (j%2 != 0)  ////GTC first solution gives different coefficient order, use i instead of j
  if (i%2 == 0)
  {
    m = -m;
  }

  //debug: Test the result for consistency
  //unsigned int j2=getSingleIndex(n,m);
  //if (j2 != j)
  //  trace_.out("getNM: Error in calculation of n,m: %i != %i\n",j2,j);
}


template<typename T>
void Zernikes<T>::setCoef(const unsigned int& zernikeIndex, const double& zernikeCoef)
{
  zernikeCoefs_[zernikeIndex] = zernikeCoef;
}


template<typename T>
double Zernikes<T>::getCoef(const unsigned int& zernikeIndex)
{
  //throws an exception if key does not exist
  return zernikeCoefs_.at(zernikeIndex);
}
