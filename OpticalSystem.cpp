/*
 * OpticalSystem.cpp
 *
 *  Created on: Jan 31, 2014
 *      Author: dailos
 */

#include "OpticalSystem.h"
#include "PDTools.h"
#include "CustomException.h"

OpticalSystem::OpticalSystem()
{
  pupilRadious_ = 0.0;
}

OpticalSystem::OpticalSystem(const cv::Mat& phase, const cv::Mat& amplitude)
{
  compute_OTF_(phase, amplitude, otf_);
}

OpticalSystem::~OpticalSystem()
{
  // TODO Auto-generated destructor stub
}

void OpticalSystem::compute_OTF_(const cv::Mat& phase, const cv::Mat& amplitude, cv::Mat& otf)
{
  //amplitude: mask defining the pupil
  //phase: aberration fase for the image
  //Consider case with only real values (only one channel)
  
  if (phase.channels() == 1 && amplitude.channels() == 1 && phase.size() == amplitude.size() && phase.type() == amplitude.type())
  {
    compute_GeneralizedPupilFunction_(phase, amplitude, generalizedPupilFunction_);
    
    cv::Mat unnormalizedOTF;
    bool full(false), corr(true);
    convolveDFT(generalizedPupilFunction_, generalizedPupilFunction_, unnormalizedOTF, corr, full);
    fftShift(unnormalizedOTF);

    //Normalize to be have 1 at oringen of the otf
    otfNormalizationFactor_ = unnormalizedOTF.col(0).row(0).clone();
    cv::Mat mfactor = cv::repeat(otfNormalizationFactor_, unnormalizedOTF.rows, unnormalizedOTF.cols);
    divSpectrums(unnormalizedOTF, mfactor, otf, cv::DFT_COMPLEX_OUTPUT);
  }
  else
  {
    throw CustomException("computeOTF_: Unsuported image type, must be both single channel.");
  }
}


void OpticalSystem::compute_GeneralizedPupilFunction_(const cv::Mat& phase, const cv::Mat& amplitude, cv::Mat& generalizedPupilFunction)
{
  //CAUTION it has to be done generic type
  //check that amplitude and phase are same size and CV_32F type, only single channel allowed
  if (phase.channels() == 1 && amplitude.channels() == 1 && phase.size() == amplitude.size() && phase.type() == amplitude.type())
  {
    cv::Mat cosPhase(phase.size(), phase.type()), sinPhase(phase.size(), phase.type());
    auto itCos = cosPhase.begin<double>();
    auto itSin = sinPhase.begin<double>();
    for(auto it = phase.begin<double>(), itEnd = phase.end<double>(); it != itEnd; ++it)
    {
      (*itSin) = std::sin(*it);
      (*itCos) = std::cos(*it);
      itSin++;
      itCos++;
    }
    
    cv::mulSpectrums(makeComplex(amplitude), makeComplex(cosPhase, sinPhase),generalizedPupilFunction,cv::DFT_COMPLEX_OUTPUT);
  }
  else
  {
    throw CustomException("computeGeneralizedPupilFunction_: Unsuported type, must be single channel.");
  }
}

cv::Mat OpticalSystem::generalizedPupilFunction() const
{
  //cv::Mat mfactor = cv::repeat(otfNormalizationFactor_, generalizedPupilFunction_.rows, generalizedPupilFunction_.cols);
  //cv::Mat normGeneralizedPupilFunction;
  //divSpectrums(generalizedPupilFunction_, mfactor, normGeneralizedPupilFunction, cv::DFT_COMPLEX_OUTPUT);
  return generalizedPupilFunction_;
}

