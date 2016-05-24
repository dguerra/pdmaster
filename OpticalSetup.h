/*
 * OpticalSetup.h
 *
 *  Created on: Feb 19, 2014
 *      Author: dailos
 */

#ifndef OPTICALSETUP_H_
#define OPTICALSETUP_H_

class OpticalSetup
{
public:
  OpticalSetup();
  OpticalSetup(const unsigned long& imageSize);
  virtual ~OpticalSetup();
  double pupilRadiousPixels()const{return pupilRadiousPixels_;};
  double cutoffPixel()const{return cutoffPixel_;};
  double k() const {return k_;};
  double diversity_ptp() const {return diversity_ptp_;};
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
  double diversity_ptp_;
};
#endif /* OPTICALSETUP_H_ */

