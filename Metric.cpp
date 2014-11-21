/*
 * Metric.cpp
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */
#include "Metric.h"
#include "PDTools.h"
#include "TelescopeSettings.h"
#include "CustomException.h"
#include "Zernikes.h"



//To bind the arguments of the member function "hello", within "object"
//Metric m;
//auto f = std::bind(&Metric::metric_from_coefs, &m, std::placeholders::_1);

Metric::Metric()
{
  // TODO Auto-generated constructor stub
}

Metric::~Metric()
{
  // TODO Auto-generated destructor stub
}

void Metric::characterizeOpticalSystem(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, std::vector<OpticalSystem>& OS)
{
  //check input dimensions
  if(coeffs.cols != 1) throw CustomException("Wrong input dimensions");
  
  TelescopeSettings tsettings(D.front().size().width);
  unsigned int pupilSideLength = optimumSideLength(D.front().size().width/2, tsettings.pupilRadiousPixels());

  unsigned int K = D.size();   //number of images to use in the algorithm
  unsigned int M = zernikeBase.size();   //number of zernike coefficients to use in the representation of each image phase
  if(K * M != coeffs.total()) throw CustomException("Coeffcient vector should contain K*M elements.");
    
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());
  
  for(unsigned int k=0; k<K; ++k)
  {  //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
    cv::Mat pupilPhase_i = Zernikes<double>::phaseMapZernikeSum(pupilSideLength,tsettings.pupilRadiousPixels(), coeffs(cv::Range(k*M, k*M + M), cv::Range::all()));
    OS.push_back(OpticalSystem(pupilPhase_i, pupilAmplitude));  //Characterized optical system
  }
   
}

void Metric::computeQ(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                      const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& Q)
{
  //double gamma_obj(0.0);  //We are not going to consider this gamma value so we set to zero
  if(OS.empty()) characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  //unsigned int K = OS.size();
  //unsigned int M = zernikeBase.size();
  
  //We use 'depth' here instead of 'type' because we want 1-channel image
  Q = cv::Mat::zeros(D.front().size(), D.front().depth());
  for(unsigned int k = 0; k < OS.size(); ++k)
  {
    cv::Mat absSj = absComplex(OS.at(k).otf());
    shift(absSj, absSj, absSj.cols/2, absSj.rows/2);
    double nc = std::sqrt(1.0/meanPowerNoise.at(k));
    //in case of undersampling optimumSideLength is bigger then image size
    cv::accumulateSquare(selectCentralROI(nc * absSj, D.front().size()), Q);
    //Q += absSj.mul(absSj);
  }
  
  //multiply all terms by first meanPowerNoise to make it equivalent to old version
  Q *= meanPowerNoise.front();
  
  Q.setTo(1.0e-35, Q < 1.0e-35);    //Avoid division by zero, before inversion
  
  TelescopeSettings tsettings(D.front().cols);
  Q.setTo(0, Zernikes<double>::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::computeAccSjDj(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                            const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& accSjDj)
{
  if(OS.empty()) characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  accSjDj = cv::Mat::zeros(D.front().size(), D.front().type());

  for(unsigned int k = 0; k < D.size(); ++k)
  {
    cv::Mat SjDj;
    cv::Mat conjSj = conjComplex(OS.at(k).otf());
    shift(conjSj, conjSj, conjSj.cols/2, conjSj.rows/2);
    cv::mulSpectrums(selectCentralROI(conjSj, D.front().size()), 
                     D.at(k), SjDj, cv::DFT_COMPLEX_OUTPUT);
    accSjDj += SjDj * (1.0 / meanPowerNoise.at(k));
  }
  
  accSjDj *= meanPowerNoise.front();
}

void Metric::noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                 const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, cv::Mat& filter)
{

  const double filter_upper_limit(1.0);
  const double filter_lower_limit(0.1);
  if(accSjDj_.empty()) computeAccSjDj(coeffs, D, zernikeBase, meanPowerNoise, OS_, accSjDj_);
  cv::Mat absAccSjDj = absComplex(accSjDj_);
  
  if(Q_.empty()) computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS_, Q_);
  cv::Mat frac;
  cv::divide(Q_, absAccSjDj.mul(absAccSjDj), frac);  //Both Q and absAccSjDj are single channel images, real matrices
  
  //These two inversion are needed to have the exact filter as in the old code
  cv::pow(frac, -1, frac);
  
  cv::blur(frac, frac, cv::Size(3,3));

  cv::pow(frac, -1, frac);
  
  cv::Mat fracFlip;
  cv::flip(frac, fracFlip, -1); //flipCode => -1 < 0 means two axes flip
  shift(fracFlip, fracFlip, 1, 1);  //shift matrix => 1 means one pixel to the right  
  
  filter = 1.0 - (meanPowerNoise.front() * (frac + fracFlip)/2.0);
  
  //remove peaks
  filter.setTo(0, filter < filter_lower_limit);
  filter.setTo(filter_upper_limit, filter > filter_upper_limit);
  
  
  //To zero-out frequencies beyond cutoff
  TelescopeSettings tsettings(D.front().cols);
  filter.setTo(0, Zernikes<double>::phaseMapZernike(1, filter.cols, tsettings.cutoffPixel()) == 0);
  
  //select only the central lobe of the filter when represented in the frequency domain
  // Find total markers
  std::vector<std::vector<cv::Point> > contours;
  //cv::Mat binary = H_ > 0;
  cv::Mat markers = cv::Mat::zeros(filter.size(), CV_8U);

  cv::findContours(cv::Mat(filter > 0), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  auto contsBegin = contours.cbegin();
  for (auto conts = contsBegin, contsEnd = contours.cend(); conts != contsEnd; ++conts)
  {
    bool calcDistance(false);
    if(cv::pointPolygonTest(*conts, cv::Point(filter.cols/2, filter.rows/2), calcDistance) > 0)
    {
      cv::drawContours(markers, contours, std::distance(contsBegin, conts), cv::Scalar::all(255), -1);
      break;
    }
  }

  filter.setTo(0, markers == 0);
  cv::blur(filter, filter, cv::Size(9,9));
}

double Metric::objectiveFunction( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                                  const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise )
{ 
  if( coeffs_.empty() || !cv::checkRange(coeffs-coeffs_, true, nullptr, 0.0, 0.0) )
  { //if a new set of coefficients are used, clean up quantities
    accSjDj_.release();
    Q_.release();
    OS_.clear();
  }
  
  coeffs.copyTo(coeffs_);
  
  if(OS_.empty()) characterizeOpticalSystem(coeffs_, D, zernikeBase, OS_);
  if(F_.empty()) objectEstimate(coeffs_, D, zernikeBase, meanPowerNoise);
  if(H_.empty()) noiseFilter(coeffs_, D, zernikeBase, meanPowerNoise, H_);
  if(accSjDj_.empty()) computeAccSjDj(coeffs_, D, zernikeBase, meanPowerNoise, OS_, accSjDj_);
  if(Q_.empty()) computeQ(coeffs_, D, zernikeBase, meanPowerNoise, OS_, Q_);
 
  
  /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 }
//  cv::Mat FH;
//  cv::mulSpectrums(F_, H_, FH, cv::DFT_COMPLEX_OUTPUT);
//  cv::Mat accDiff = cv::Mat::zeros(FH.size(), FH.depth());
//  for(unsigned int k = 0; k<OS_.size(); ++k)
//  {
//    cv::Mat DjH;
//    cv::mulSpectrums(D.at(k), makeComplex(H_), DjH, cv::DFT_COMPLEX_OUTPUT);
//    cv::Mat FHSj;
//    cv::Mat Sj = OS_.at(k).otf();
//    shift(Sj, Sj, Sj.cols/2, Sj.rows/2);
//    cv::mulSpectrums(FH, selectCentralROI(Sj, FH.size()), FHSj, cv::DFT_COMPLEX_OUTPUT);  
//    accDiff += absComplex(DjH - FHSj);
//  }
//  std::cout << "L new, old style: " << cv::sum(accDiff) << std::endl;
  
  cv::Mat absAccSjDj = absComplex(accSjDj_);
  cv::Mat num, frac, accDj = cv::Mat::zeros(Q_.size(), Q_.type());
  cv::multiply(absAccSjDj, absAccSjDj, num);
  cv::divide(num, Q_, frac);
  
  for(cv::Mat Dj : D)
  {
    cv::Mat absDj = absComplex(Dj);
    //they're real matrices, so we use "mul" instead of mulSpectrums
    cv::accumulateSquare(absDj, accDj);  //Accumulate the square of the quantity absDj
    //accDj += absDj.mul(absDj);   
  }
  L_ = cv::sum(accDj - frac).val[0];
  return L_;
}
 
void Metric::objectEstimate(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                            const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise)
{
  if( coeffs_.empty() || !cv::checkRange(coeffs-coeffs_, true, nullptr, 0.0, 0.0) )
  { //if a new set of coefficients are used, clean up quantities
    accSjDj_.release();
    Q_.release();
    OS_.clear();
  }
  
  coeffs.copyTo(coeffs_);
  if(accSjDj_.empty()) computeAccSjDj(coeffs_, D, zernikeBase, meanPowerNoise, OS_, accSjDj_);
  if(Q_.empty()) computeQ(coeffs_, D, zernikeBase, meanPowerNoise, OS_, Q_);
  cv::Mat Q_1;  //Inverse of every element of Q_
  cv::pow(Q_, -1, Q_1);
  cv::mulSpectrums(makeComplex(Q_1), accSjDj_, F_, cv::DFT_COMPLEX_OUTPUT);
  
  //Filter the result out
  if(H_.empty()) noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, H_);
  std::cout << "filter.new.(80,80): " <<  H_.at<double>(80,80) << std::endl;
  //cv::mulSpectrums(makeComplex(H_), F_, F_, cv::DFT_COMPLEX_OUTPUT);
}
  
  
cv::Mat Metric::finiteDifferencesGradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                               const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise)
{
  double EPS(1.0e-8);
	cv::Mat xh = coeffs.clone();
	double fold = objectiveFunction(coeffs, D, zernikeBase, meanPowerNoise);
  
  cv::Mat temp = coeffs.clone();
  cv::Mat h = EPS * cv::abs(temp);
  h.setTo(EPS, h == 0.0);
  xh = temp + h;
  h = xh - temp;
  double fh = objectiveFunction(xh, D, zernikeBase, meanPowerNoise);
  xh = temp.clone();
  cv::Mat df;
  cv::divide((fh - fold), h, df);
  
  return df;
}


cv::Mat Metric::gradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise)
{
  if( coeffs_.empty() || !cv::checkRange(coeffs-coeffs_, true, nullptr, 0.0, 0.0) )
  {
    //if a new set of coefficients are used, clean up quantities
    accSjDj_.release();
    Q_.release();
    OS_.clear();
  }
  coeffs.copyTo(coeffs_);
  
  /////////////////////////////////////////////////
  //Compute gradient vector, g, with N = K*M elements
  if(F_.empty()) objectEstimate(coeffs, D, zernikeBase, meanPowerNoise);
    
  unsigned int K = OS_.size();
  unsigned int M = zernikeBase.size();
  cv::Mat conjF  = conjComplex(F_);
  cv::Mat absF   = absComplex(F_);
  cv::Mat absF2  = absF.mul(absF);

  g_ = cv::Mat::zeros(K*M, 1, cv::DataType<double>::type);

  for(unsigned int k = 0; k < OS_.size(); ++k)
  {
    cv::Mat P, pj, FDj, F2Sj, re, pjre, pjre_f, term;

    P = OS_.at(k).generalizedPupilFunction();
    cv::idft(P, pj, cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(conjF, D.at(k), FDj, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(absF2, OS_.at(k).otf(), F2Sj, cv::DFT_COMPLEX_OUTPUT);
    cv::idft(FDj-F2Sj, re, cv::DFT_REAL_OUTPUT);   //this term turn out to be real after the inverse fourier transform
    cv::multiply(pj, re, pjre);   //both pj and re are real matrices, so multiply in the other way

    cv::dft(pjre, pjre_f, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    cv::mulSpectrums(conjComplex(P), pjre_f, term, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat acc = -2 * std::sqrt(K) * splitComplex(term).second;   //only takes imaginary part of the term

    for(unsigned int m = 0; m < M; ++m)
    {
      if(!zernikeBase.at(m).empty())
      {
        g_.at<double>((k * M) + m, 0) = acc.dot(zernikeBase.at(m));
      }
      else
      {
        g_.at<double>((k * M) + m, 0) = 0.0;
      }
    }
  }
  return g_;

}
