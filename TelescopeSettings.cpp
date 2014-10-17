/*
 * TelescopeSettings.cpp
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */

#include "TelescopeSettings.h"
TelescopeSettings::TelescopeSettings()
{
  workingLambda_ = 5250.6e-10; //(meters)
  pixelSizeArcsec_ = 0.055;
  telescopeDiameter_ = 1.0; //(meters)
}

TelescopeSettings::TelescopeSettings(const unsigned long& imageSize)
{
  workingLambda_ = 5250.6e-10; //(meters)
  pixelSizeArcsec_ = 0.055;
  telescopeDiameter_ = 1.0; //(meters)
  imageSize_ = imageSize;
  double pixelSizeRadians = (pixelSizeArcsec_ * 3.141516) / (180.0 * 3600.0);
  double cutoffFrequencyRadians = telescopeDiameter_ / workingLambda_;
  double sampleIntervalRadians = 1 / (imageSize_ * pixelSizeRadians);
  pupilRadiousPixels_ = cutoffFrequencyRadians / (2 * sampleIntervalRadians);
  unsigned long cutoffPixelFineTuning = 1.0; //Constant to decrese cutoff effective value
  cutoffPixel_ = 2 * pupilRadiousPixels_ - cutoffPixelFineTuning;
}

TelescopeSettings::~TelescopeSettings()
{
  // TODO Auto-generated destructor stub
}

