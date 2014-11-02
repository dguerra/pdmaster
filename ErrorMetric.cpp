/*
 * ErrorMetric.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: dailos
 */

#include <iostream>

#include "Zernikes.h"
#include "PDTools.h"
#include "ErrorMetric.h"
#include "CustomException.h"
#include "TelescopeSettings.h"
#include "NoiseEstimator.h"
#include "NoiseFilter.h"


//Gonsalves objective function
//www.google.com

//Other names could be PhaseDiversityParameters, PhaseDiversityCharacterization, PhaseDiversityValues, PhaseDiversityMagnitudes

ErrorMetric::ErrorMetric()
{
  // TODO Auto-generated constructor stub

}

ErrorMetric::ErrorMetric( const OpticalSystem& focusedOS, const OpticalSystem& defocusedOS,
                          const cv::Mat& D0, const cv::Mat& Dk, const double& meanPowerNoiseD0, 
                          const double& meanPowerNoiseDk, const std::map<unsigned int, cv::Mat>&
                          zernikeCatalog, const cv::Mat& zernikesInUse, cv::Mat& eCoreZeroMean,
                          std::vector<cv::Mat>& dedcCoreZeroMean)
{
  TelescopeSettings tsettings(D0.cols);

  cv::Mat T0 = focusedOS.otf();
  shift(T0,T0,T0.cols/2, T0.rows/2);
  T0.setTo(0, Zernikes<double>::phaseMapZernike(1, T0.cols, tsettings.cutoffPixel()) == 0);
  cv::Mat T0_cropped = takeoutImageCore(T0, D0.cols);

  cv::Mat Tk = defocusedOS.otf();
  shift(Tk,Tk,Tk.cols/2, Tk.rows/2);
  Tk.setTo(0, Zernikes<double>::phaseMapZernike(1, Tk.cols, tsettings.cutoffPixel()) == 0);
  cv::Mat Tk_cropped = takeoutImageCore(Tk, D0.cols);

  compute_Q_(T0_cropped, Tk_cropped, meanPowerNoiseD0/meanPowerNoiseDk, Q_);
  cv::Mat Q = makeComplex(Q_);

  cv::Mat Q2 = makeComplex(Q_.mul(Q_));
  compute_FM_(T0_cropped, Tk_cropped, D0, Dk, meanPowerNoiseD0/meanPowerNoiseDk, Q2, FM_);

  //Create noise filter here to use later on
  NoiseFilter filter(T0_cropped, Tk_cropped, D0, Dk, Q_.mul(Q_), meanPowerNoiseD0, meanPowerNoiseDk);
  noiseFilter_ = makeComplex(filter.H());
  cv::Mat D0H, DkH;
  cv::mulSpectrums(D0, noiseFilter_, D0H, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(Dk, noiseFilter_, DkH, cv::DFT_COMPLEX_OUTPUT);

  compute_E_(T0_cropped, Tk_cropped, D0H, DkH, Q, E_);

  compute_dTdc_(focusedOS, zernikeCatalog, zernikesInUse, dT0dc_);
  compute_dTdc_(defocusedOS, zernikeCatalog, zernikesInUse, dTkdc_);

  std::vector<cv::Mat> dT0dc_cropped, dTkdc_cropped;

/*
  auto shiftAndCrop = [&] (const cv::Mat& src) -> cv::Mat
  {
    cv::Mat dst;
    if(!src.empty())
    {
      cv::Mat srcShift(src);
      shift(srcShift, srcShift, srcShift.cols/2, srcShift.rows/2);
      cv::Mat dst = takeoutImageCore(srcShift, D0.cols);
    }
    return dst;
  };
*/

  for(cv::Mat dT0dci : dT0dc_)
  {
    if(!dT0dci.empty())
    {
      shift(dT0dci, dT0dci, dT0dci.cols/2, dT0dci.rows/2);
      cv::Mat dT0dciCore = takeoutImageCore(dT0dci, D0.cols);
      dT0dc_cropped.push_back(dT0dciCore);
    }
    else
    {
      dT0dc_cropped.push_back(cv::Mat());
    }
  }
  for(cv::Mat dTkdci : dTkdc_)
  {
    if(!dTkdci.empty())
    {
      shift(dTkdci, dTkdci, dTkdci.cols/2, dTkdci.rows/2);
      cv::Mat dTkdciCore = takeoutImageCore(dTkdci, D0.cols);
      dTkdc_cropped.push_back(dTkdciCore);
    }
    else
    {
      dTkdc_cropped.push_back(cv::Mat());
    }

  }

  compute_dEdc_(T0_cropped, Tk_cropped, D0H, DkH, Q, Q2, dT0dc_cropped, dTkdc_cropped, meanPowerNoiseD0/meanPowerNoiseDk, dEdc_);
  
  cv::Mat fm;
  //showRestore(EM, fm);
  std::cout << "Total restored image energy: " << cv::sum(fm) << std::endl;
  unsigned int imageCoreSize_ = 70;
  eCoreZeroMean = backToImageSpace(E_, cv::Size(imageCoreSize_, imageCoreSize_));
  for(cv::Mat dEdci : dEdc_) dedcCoreZeroMean.push_back(backToImageSpace(dEdci, cv::Size(imageCoreSize_, imageCoreSize_)));

}

cv::Mat ErrorMetric::backToImageSpace(const cv::Mat& fourierSpaceMatrix, const cv::Size& centralROI)
{
  cv::Mat imageMatrixROIZeroMean;
  if(!fourierSpaceMatrix.empty())
  {
    cv::Mat imageMatrix;
    cv::Mat fourierSpaceMatrixShift(fourierSpaceMatrix);
    //shift quadrants back to origin in the corner, inverse transform, take central region, force zero-mean
    shift(fourierSpaceMatrixShift, fourierSpaceMatrixShift, fourierSpaceMatrixShift.cols/2, fourierSpaceMatrixShift.rows/2);
    cv::idft(fourierSpaceMatrixShift, imageMatrix, cv::DFT_REAL_OUTPUT);
    cv::Mat imageMatrixROI = takeoutImageCore(imageMatrix, centralROI.height);
    imageMatrixROIZeroMean = imageMatrixROI - cv::mean(imageMatrixROI);
  }
  return imageMatrixROIZeroMean;
}

ErrorMetric::~ErrorMetric()
{
  // TODO Auto-generated destructor stub
}

void ErrorMetric::compute_FM_(const cv::Mat& T0, const cv::Mat& Tk,
                              const cv::Mat& D0, const cv::Mat& Dk, const double& gamma, const cv::Mat& Q2, cv::Mat& FM)
{
  cv::Mat D0T0, DkTk;
  cv::mulSpectrums(D0,conjComplex(T0),D0T0, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(Dk,conjComplex(Tk),DkTk, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(Q2, D0T0+(gamma*DkTk), FM, cv::DFT_COMPLEX_OUTPUT);
}

//F represents the object estimate
void ErrorMetric::computeObjectEstimate_(const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& S, const double& gamma, cv::Mat& F, cv::Mat& Q)
{
  //Compute first Q value, needed to know the object estimate
  Q = cv::Mat::zeros(S.front().size(), S.front().type());
  for(cv::Mat Sj : S)
  {
    cv::Mat absSj = absComplex(Sj);
    Q += absSj.mul(absSj);
  }
  Q = Q + gamma;

  //Compute now the object estimate, using Q
  cv::Mat acc = cv::Mat::zeros(D.front().size(), D.front().type());

  for(unsigned int k = 0; k < D.size(); ++k)
  {
    cv::Mat SjDj;
    cv::mulSpectrums(conjComplex(S.at(k)), D.at(k), SjDj, cv::DFT_COMPLEX_OUTPUT);
    acc += SjDj;
  }
  cv::Mat Q_1;
  cv::pow(Q, -1.0, Q_1);  //Q is a real matrix
  cv::mulSpectrums(makeComplex(Q_1), acc, F, cv::DFT_COMPLEX_OUTPUT);
}

void ErrorMetric::compute_Q_(const cv::Mat& T0, const cv::Mat& Tk, const double& gamma, cv::Mat& Q)
{
  cv::Mat absT0 = absComplex(T0);
  cv::Mat absTk = absComplex(Tk);
  double qTuning_(0.0);   //additive constant for Q, adding offset, (not needed by now)
  Q = absT0.mul(absT0) + gamma * absTk.mul(absTk) + qTuning_;
  //Suppress very small values to avoid peaks after dividing by one?: //1/(sqrt(tmp)>1.0e-35)*tsupport  ????
  cv::sqrt(Q, Q);
  Q.setTo(1.0e-35, Q<1.0e-35);  //CAUTION! I need an explanation!
  cv::pow(Q, -1.0, Q);   //Q is a real matrix

  TelescopeSettings tsettings(T0.cols);
  Q.setTo(0, Zernikes<double>::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
}

void ErrorMetric::compute_E_(const cv::Mat& T0, const cv::Mat& Tk,
                             const cv::Mat& D0, const cv::Mat& Dk, const cv::Mat& Q, cv::Mat& E)
{
  cv::Mat DkT0, D0Tk;

  cv::mulSpectrums(Dk, T0, DkT0, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(D0, Tk, D0Tk, cv::DFT_COMPLEX_OUTPUT);  //DFT_SCALE?? DFT_COMPLEX_OUTPUT??

  cv::mulSpectrums(Q, (DkT0 - D0Tk), E, cv::DFT_COMPLEX_OUTPUT);
}

void ErrorMetric::compute_dTdc_(const OpticalSystem& os, const std::map<unsigned int, cv::Mat>& zernikeCatalog,
                                const cv::Mat& zernikesInUse, std::vector<cv::Mat>& dTdc)
{
  dTdc.clear();  //Clear all elements first
  for(unsigned int currentIndex = 1; currentIndex <= zernikesInUse.total(); ++currentIndex)
  {
    cv::Mat dTdci;
    if(zernikesInUse.at<bool>(currentIndex-1, 0) == true)
    {
      cv::Mat ZH;

      cv::mulSpectrums(makeComplex(zernikeCatalog.at(currentIndex)), os.generalizedPupilFunction(), ZH, cv::DFT_COMPLEX_OUTPUT);
      //cv::mulSpectrums(makeComplex(zr), os.generalizedPupilFunction(), ZH, cv::DFT_COMPLEX_OUTPUT);
      cv::Mat H_ZH = divComplex(crosscorrelation(os.generalizedPupilFunction(), ZH), os.otfNormalizationFactor());
      //showComplex(H_ZH, "H_ZH", false, false);
      cv::Mat H_ZHFlipped;

      cv::flip(H_ZH, H_ZHFlipped, -1); //flipCode => -1 < 0 means two axes flip
      shift(H_ZHFlipped, H_ZHFlipped, 1, 1);  //shift matrix => 1,1 means one pixel to the right, one pixel down
      std::pair<cv::Mat, cv::Mat> splitComplexMatrix = splitComplex(H_ZH - conjComplex(H_ZHFlipped));
      dTdci = makeComplex((-1)*splitComplexMatrix.second, splitComplexMatrix.first);//equivalent to multiply by imaginary unit i
    }
    dTdc.push_back(dTdci);  //push_back empty matrix if zernike index is not in use
  }
}

void ErrorMetric::compute_dEdc_( const cv::Mat& T0, const cv::Mat& Tk,
                                 const cv::Mat& D0, const cv::Mat& Dk, const cv::Mat& Q, const cv::Mat& Q2,
                                 const std::vector<cv::Mat>& dT0dc, const std::vector<cv::Mat>& dTkdc, const double& gamma,
                                 std::vector<cv::Mat>& dEdc )
{
  if(dT0dc.size() == dTkdc.size())
  {
    dEdc.clear();  //clear all elements first
    cv::Mat T0dT0dci, TkdTkdci;

    for(auto dT0dci = dT0dc.cbegin(), dT0dciEnd = dT0dc.cend(),
             dTkdci = dTkdc.cbegin(), dTkdciEnd = dTkdc.cend();
             dT0dci != dT0dciEnd, dTkdci != dTkdciEnd; ++dT0dci, ++dTkdci)
    {
      cv::Mat dEdci;
      if(!(*dT0dci).empty() && !(*dTkdci).empty())
      {
        cv::mulSpectrums(conjComplex(T0), *dT0dci, T0dT0dci, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(conjComplex(Tk), *dTkdci, TkdTkdci, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat rTerm = T0dT0dci + gamma * TkdTkdci;

        cv::Mat realTerm = splitComplex(rTerm).first;

        cv::Mat D0Q, DkQ;
        cv::mulSpectrums(D0, Q, D0Q, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(Dk, Q, DkQ, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat Q2T0, Q2Tk;
        cv::mulSpectrums(Q2, T0, Q2T0, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(Q2, Tk, Q2Tk, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat Q2T0RealTerm, Q2TkRealTerm;
        cv::mulSpectrums(Q2T0,makeComplex(realTerm),Q2T0RealTerm, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(Q2Tk,makeComplex(realTerm),Q2TkRealTerm, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat leftTerm, rightTerm;
        cv::mulSpectrums(DkQ, (*dT0dci)-Q2T0RealTerm, leftTerm, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(D0Q, (*dTkdci)-Q2TkRealTerm, rightTerm, cv::DFT_COMPLEX_OUTPUT);
        dEdci = leftTerm-rightTerm;
      }
      dEdc.push_back(dEdci); //push_back empty matrix if dT0dci or dTkdci are empty
    }
  }
  else
  {
    throw CustomException("ErrorMetric: Vectors dT0dc and dTkdc must be equal in size.");
  }
}
