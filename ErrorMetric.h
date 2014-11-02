/*
 * ErrorMetric.h
 *
 *  Created on: Nov 13, 2013
 *      Author: dailos
 */

#ifndef ERRORMETRIC_H_
#define ERRORMETRIC_H_

#include <vector>

#include "opencv2/opencv.hpp"
#include "OpticalSystem.h"

class ErrorMetric
{
public:
  ErrorMetric( const OpticalSystem& focusedOS, const OpticalSystem& defocusedOS, const cv::Mat& D0,
               const cv::Mat& Dk, const double& meanPowerNoiseD0, const double& meanPowerNoiseDk,
               const std::map<unsigned int, cv::Mat>& zernikeCatalog, const cv::Mat& zernikesInUse,
               cv::Mat& eCoreZeroMean, std::vector<cv::Mat>& dedcCoreZeroMean);
  virtual ~ErrorMetric();
  cv::Mat backToImageSpace(const cv::Mat& fourierSpaceMatrix, const cv::Size& centralROI);
  cv::Mat E()const {return E_;};
  std::vector<cv::Mat> dEdc()const {return dEdc_;};
  cv::Mat FM()const {return FM_;};
  cv::Mat noiseFilter()const{return noiseFilter_;};

private:
  ErrorMetric();  //Private default constructor
  void compute_Q_(const cv::Mat& T0, const cv::Mat& Tk, const double& gamma, cv::Mat& Q);
  void computeObjectEstimate_(const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& S, const double& gamma, cv::Mat& F, cv::Mat& Q);
  void compute_E_(const cv::Mat& T0, const cv::Mat& Tk,
                  const cv::Mat& D0, const cv::Mat& Dk, const cv::Mat& Q, cv::Mat& E);
  void compute_dTdc_(const OpticalSystem& os, const std::map<unsigned int, cv::Mat>& zernikeCatalog, const cv::Mat& zernikesInUse, std::vector<cv::Mat>& dTdc);

  void compute_dEdc_(const cv::Mat& T0, const cv::Mat& Tk,
                     const cv::Mat& D0, const cv::Mat& Dk, const cv::Mat& Q, const cv::Mat& Q2,
                     const std::vector<cv::Mat>& dT0dc, const std::vector<cv::Mat>& dTkdc, const double& gamma,
                     std::vector<cv::Mat>& dEdc );

  void compute_FM_(const cv::Mat& T0, const cv::Mat& Tk,
                   const cv::Mat& D0, const cv::Mat& Dk, const double& gamma, const cv::Mat& Q2, cv::Mat& FM);

  cv::Mat Q_;
  cv::Mat E_;
  cv::Mat FM_;   //Optimum restored scene

  std::vector<cv::Mat> dT0dc_;
  std::vector<cv::Mat> dTkdc_;
  std::vector<cv::Mat> dEdc_;

  cv::Mat noiseFilter_;

};
#endif /* ERRORMETRIC_H_ */
