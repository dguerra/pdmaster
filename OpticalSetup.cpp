/*
 * OpticalSetup.cpp
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */
 //Rename as OpticalSetup
 
#include <cmath>
#include "OpticalSetup.h"
#include <iostream>
OpticalSetup::OpticalSetup()
{
workingLambda_ = 4.69600e-7;
pixelSizeArcsec_ = 0.072;
telescopeDiameter_ = 0.475; //(meters)
  
  defocusTranslationAlongOpticalAxis_ = 0.0082;  //(meters)
  focalLength_ = 0.475;  //(meters)
}

OpticalSetup::OpticalSetup(const unsigned long& imageSize)
{
workingLambda_ = 4.69600e-7;
pixelSizeArcsec_ = 0.072;
telescopeDiameter_ = 0.475; //(meters)


  defocusTranslationAlongOpticalAxis_ = 0.0082;  //(meters)
  focalLength_ = 22.35;  //(meters)
  imageSize_ = imageSize;
  double pixelSizeRadians = (pixelSizeArcsec_ * 3.141516) / (180.0 * 3600.0);
  double cutoffFrequencyRadians = telescopeDiameter_ / workingLambda_;   //Angular cutoff frequency
  double sampleIntervalRadians = 1.0 / (imageSize_ * pixelSizeRadians);
  pupilRadiousPixels_ = cutoffFrequencyRadians / (2.0 * sampleIntervalRadians);
  unsigned long cutoffPixelFineTuning = 1.0; //Constant to decrese cutoff effective value
  cutoffPixel_ = 2.0 * pupilRadiousPixels_; // - cutoffPixelFineTuning;
  const double pi = 3.14159265359;
  //k == 1 means phase shift at the edge of the aperture equal to pi radians or 1/2 waves (two times the diversity peak to peak)
  k_ = (-1.0) * std::pow(telescopeDiameter_/focalLength_,2.0) * defocusTranslationAlongOpticalAxis_/(workingLambda_ * 4.0);
  diversity_ptp_ = 0.985;
}

OpticalSetup::~OpticalSetup()
{
  // TODO Auto-generated destructor stub
}


/*
//Mats setup:
workingLambda_ = 4.69600e-7;
pixelSizeArcsec_ = 0.072;
telescopeDiameter_ = 0.475; //(meters)
a4_ = -1.97;

//Joseantonio setup:
workingLambda_ = 5250.6e-10; //(meters)
pixelSizeArcsec_ = 0.055;
telescopeDiameter_ = 1.0; //(meters)
a4_ = -2.21209;
*/

