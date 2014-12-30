/*
 * GetStep.cpp
 *
 *  Created on: Oct 30, 2014
 *      Author: dailos
 */
#include <iostream>
#include "ErrorMetric.h"
#include "Metric.h"
#include "Optimization.h"
#include "GetStep.h"
//#include "PDTools.h"
#include "Zernikes.h"
#include "Zernikes.cpp"



int getstep(cv::Mat& c, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& diversityPhase, const cv::Mat& pupilAmplitude,
		const unsigned int& pupilSideLength, const cv::Mat& zernikesInUse, const cv::Mat& alignmentSetup,
		const std::map<unsigned int, cv::Mat>& zernikeCatalog,
		const double& pupilRadiousP, const std::vector<double>& meanPowerNoise, double& lmPrevious,
		unsigned int& numberOfNonSingularities, double& singularityThresholdOverMaximum, cv::Mat& dc)
{
    cv::Mat offsetPupilPhase = Zernikes<double>::phaseMapZernikeSum(pupilSideLength, pupilRadiousP, c.setTo(0, 1-zernikesInUse ));

    std::vector<cv::Mat> pupilPhase;
    for(cv::Mat diversityPhase_i : diversityPhase)
    {
      pupilPhase.push_back(offsetPupilPhase + diversityPhase_i);
    }
    // + c2  * Zernikes<double>::phaseMapZernike(2, imgSize/2, tsettings.pupilRadiousPixels())   //tiptilt X
    // + c3  * Zernikes<double>::phaseMapZernike(3, imgSize/2, tsettings.pupilRadiousPixels())   //tiptilt Y

    std::vector<OpticalSystem> os;
    for(cv::Mat pupilPhase_i : pupilPhase)
    {
      os.push_back(OpticalSystem(pupilPhase_i, pupilAmplitude));
    }

    double filterTuning_ = 1.0;
    cv::Mat eCoreZeroMean;
    std::vector<cv::Mat> dedcCoreZeroMean;
     
    
    ErrorMetric EM( os.front(), os.back(), D.front(), D.back(), meanPowerNoise.front()*filterTuning_, 
                    meanPowerNoise.back()*filterTuning_, zernikeCatalog, zernikesInUse, eCoreZeroMean, dedcCoreZeroMean );
    
    double lmCurrent = cv::sum(eCoreZeroMean.mul(eCoreZeroMean)).val[0]/eCoreZeroMean.total();
  
    double lmIncrement = std::abs(lmCurrent - lmPrevious)/lmCurrent;

    cv::Mat c2, dc2;
    cv::pow(c.setTo(0, alignmentSetup), 2.0, c2);
    cv::pow(dc.setTo(0, alignmentSetup), 2.0, dc2);
    double cRMS = std::sqrt(cv::sum(c2).val[0]);
    double dcRMS = std::sqrt(cv::sum(dc2).val[0]);

    std::cout << "lm_: " << lmCurrent << std::endl;
    std::cout << "lmIncrement_: " << lmIncrement << std::endl;
    std::cout << "c_: " << c.t() << std::endl;
    std::cout << "cRMS_: " << cRMS << std::endl;
    std::cout << "dcRMS_: " << dcRMS << std::endl;
    std::cout << "numberOfNonSingularities_: " << numberOfNonSingularities << std::endl;
    std::cout << "singularityThresholdOverMaximum_: " << singularityThresholdOverMaximum << std::endl;

    lmPrevious = lmCurrent;

    double dcRMS_Minimum_ = 1.0e-2;
    double lmIncrement_Minimum_ = 0.0;

    if((dcRMS < dcRMS_Minimum_) || (lmIncrement < lmIncrement_Minimum_))
    {
      if(dcRMS < dcRMS_Minimum_) std::cout << "dcRMS_Minimum has been reached." << std::endl;
      if(lmIncrement < lmIncrement_Minimum_) std::cout << "lmIncrement_Minimum has been reached." << std::endl;
      return 1;
    }
    //Also add the case where tip-tilt rms are lower than minimum
    //  (std::sqrt(dc(2)*dc(2)+dc(3)*dc(3)) < dcRMS_Minimum_) ||

    //e, and dedc must be taken from central subfield and mean zero here before optimization
    Optimization optimumIncrement(eCoreZeroMean, dedcCoreZeroMean);

    numberOfNonSingularities = optimumIncrement.numberOfNonSingularities();
    singularityThresholdOverMaximum = optimumIncrement.singularityThresholdOverMaximum();

    dc = optimumIncrement.dC();

  return 0;
}

