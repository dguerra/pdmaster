/*
 * TelescopeSettings.h
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */

#ifndef TELESCOPESETTINGS_H_
#define TELESCOPESETTINGS_H_

class TelescopeSettings
{
public:
  TelescopeSettings();
  TelescopeSettings(const unsigned long& imageSize);
  virtual ~TelescopeSettings();
  double pupilRadiousPixels()const{return pupilRadiousPixels_;};
  double cutoffPixel()const{return cutoffPixel_;};
  double k() const {return k_;};
  void imageSize(const unsigned long& imgSize){imageSize_ = imgSize;};
private:
  double workingLambda_;  //(meters)
  double pixelSizeArcsec_;
  double telescopeDiameter_; //(meters)
  unsigned long imageSize_;
  double pupilRadiousPixels_;
  double cutoffPixel_;
  double defocusTranslationAlongOpticalAxis_;  // (meters)
  double focalLength_;  //(meters)
  double k_;   //RMS defocus coefficienct
};
#endif /* TELESCOPESETTINGS_H_ */

