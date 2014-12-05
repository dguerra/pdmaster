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
#include "FITS.h"


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
   
    //in case of undersampling optimumSideLength is bigger then image size
    cv::accumulateSquare(selectCentralROI(absSj, D.front().size()), Q);
    //Q += absSj.mul(absSj);
  }
  
  TelescopeSettings tsettings(D.front().cols);
  Q.setTo(0, Zernikes<double>::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::computeP(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                            const std::vector<double>& meanPowerNoise, std::vector<OpticalSystem>& OS, cv::Mat& P)
{
  if(OS.empty()) characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  P = cv::Mat::zeros(D.front().size(), D.front().type());

  //P=accumulate over J { Dj * conj(Sj) } 
  for(unsigned int k = 0; k < D.size(); ++k)
  {
    cv::Mat SjDj;
    cv::Mat conjSj = conjComplex(OS.at(k).otf());
    shift(conjSj, conjSj, conjSj.cols/2, conjSj.rows/2);
    cv::mulSpectrums(selectCentralROI(conjSj, D.front().size()), 
                     D.at(k), SjDj, cv::DFT_COMPLEX_OUTPUT);
    
    P += SjDj;
  }
}

void Metric::noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                 const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, cv::Mat& filter)
{
  const double filter_upper_limit(1.0);
  const double filter_lower_limit(0.1);
  if(P_.empty()) computeP(coeffs, D, zernikeBase, meanPowerNoise, OS_, P_);
  cv::Mat absP = absComplex(P_);
  
  if(Q_.empty()) computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS_, Q_);
  cv::Mat frac;
  cv::divide(Q_, absP.mul(absP), frac);  //Both Q and absP are single channel images, real matrices
  
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
    P_.release();
    Q_.release();
    OS_.clear();
  }
  
  coeffs.copyTo(coeffs_);
  
  if(OS_.empty()) characterizeOpticalSystem(coeffs_, D, zernikeBase, OS_);
  if(F_.empty()) objectEstimate(coeffs_, D, zernikeBase, meanPowerNoise);
  if(H_.empty()) noiseFilter(coeffs_, D, zernikeBase, meanPowerNoise, H_);
  if(P_.empty()) computeP(coeffs_, D, zernikeBase, meanPowerNoise, OS_, P_);
  if(Q_.empty()) computeQ(coeffs_, D, zernikeBase, meanPowerNoise, OS_, Q_);
  
  if(false)
  {
    /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 }
    cv::Mat accDiff = cv::Mat::zeros(F_.size(), F_.depth());
    for(unsigned int k = 0; k<OS_.size(); ++k)
    {
      cv::Mat DjH;
      cv::mulSpectrums(D.at(k), makeComplex(H_), DjH, cv::DFT_COMPLEX_OUTPUT);
      cv::Mat FHSj;
      cv::Mat Sj = OS_.at(k).otf();
      shift(Sj, Sj, Sj.cols/2, Sj.rows/2);
      cv::mulSpectrums(F_, selectCentralROI(Sj, F_.size()), FHSj, cv::DFT_COMPLEX_OUTPUT);
      cv::accumulateSquare(absComplex(DjH - FHSj), accDiff);
    }
 
    L_ = cv::sum(accDiff).val[0];
    std::cout << "L value from diff: " << L_ << std::endl;   //8.24441e-05
  }

  
  cv::Mat absP = absComplex(P_);
  cv::Mat frac, accDjH = cv::Mat::zeros(Q_.size(), Q_.type());
  
  cv::multiply(absP.mul(absP), H_.mul(H_)/Q_, frac);
  
  for(cv::Mat Dj : D)
  {
    cv::Mat DjH;
    cv::mulSpectrums(Dj, makeComplex(H_), DjH, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat absDjH = absComplex(DjH);
    //they're real matrices, so we use "mul" instead of mulSpectrums
    cv::accumulateSquare(absDjH, accDjH);  //Accumulate the square of the quantity absDj
  }
  
  L_ = cv::sum(accDjH - frac).val[0];
  std::cout << "L value from eq: " << L_ << std::endl;
  return L_;
  
}
 
void Metric::objectEstimate(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                            const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise)
{
  if( coeffs_.empty() || !cv::checkRange(coeffs-coeffs_, true, nullptr, 0.0, 0.0) )
  { //if a new set of coefficients are used, clean up quantities
    P_.release();
    Q_.release();
    OS_.clear();
  }
  
  coeffs.copyTo(coeffs_);
  if(P_.empty()) computeP(coeffs_, D, zernikeBase, meanPowerNoise, OS_, P_);
  if(Q_.empty()) computeQ(coeffs_, D, zernikeBase, meanPowerNoise, OS_, Q_);
  cv::Mat Q_1;  //Inverse of every element of Q_
  cv::pow(Q_, -1, Q_1);
  cv::mulSpectrums(makeComplex(Q_1), P_, F_, cv::DFT_COMPLEX_OUTPUT);
  
  //Filter the result out
  if(H_.empty()) noiseFilter(coeffs_, D, zernikeBase, meanPowerNoise, H_);
  cv::mulSpectrums(F_, makeComplex(H_), F_, cv::DFT_COMPLEX_OUTPUT);
  //std::cout << "filter.new.(80,80): " <<  H_.at<double>(80,80) << std::endl;
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


cv::Mat Metric::gradient(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                         const std::vector<double>& meanPowerNoise)
{
  if( coeffs_.empty() || !cv::checkRange(coeffs-coeffs_, true, nullptr, 0.0, 0.0) )
  {
    //if a new set of coefficients are used, clean up quantities
    P_.release();
    Q_.release();
    OS_.clear();
  }
  coeffs.copyTo(coeffs_);
  
  /////////////////////////////////////////////////
  //Compute gradient vector, g, with N = K*M elements
  if(F_.empty()) objectEstimate(coeffs, D, zernikeBase, meanPowerNoise);
  if(P_.empty()) computeP(coeffs_, D, zernikeBase, meanPowerNoise, OS_, P_);
  
  unsigned int J = OS_.size();
  unsigned int M = zernikeBase.size();
  
  g_ = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);

  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat Pj, pj, FDj, F2Sj, re, pjre, pjre_f, term;

    Pj = OS_.at(j).generalizedPupilFunction();
    
    cv::Mat Pj_pad = cv::Mat::zeros(D.front().size(), Pj.type());
    Pj.copyTo(selectCentralROI(Pj_pad, Pj.size()));
    std::cout << "Pj_pad.at<std::complex<double> >(50,50) before norm: " << Pj_pad.at<std::complex<double> >(50,50) << std::endl;

    cv::Mat normFactor;
    cv::resize(OS_.at(j).otfNormalizationFactor(), normFactor, Pj_pad.size());
    Pj_pad = divComplex(Pj_pad, normFactor);
    
    std::cout << "Pj_pad.at<std::complex<double> >(50,50) after norm: " << Pj_pad.at<std::complex<double> >(50,50) << std::endl;
    //!!¡¡When multiply by conjuagte, use mulSpectrums(A, B, out, cv::DFT_COMPLEX_OUTPUT, true);
    //square of absolute value of z is: |z|^2 = z * conj(z)
    
    cv::Mat pj_pad = Pj_pad.clone();
    shift(pj_pad, pj_pad, pj_pad.cols/2, pj_pad.rows/2);
    cv::idft(pj_pad, pj_pad, cv::DFT_REAL_OUTPUT);
    
    
    cv::Mat abs2P, abs2PSj;
    bool conjB(true);
    
    cv::mulSpectrums(P_, P_, abs2P, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat Sj = OS_.at(j).otf();
    shift(Sj, Sj, Sj.cols/2, Sj.rows/2);
   
    cv::mulSpectrums(abs2P, selectCentralROI(Sj, abs2P.size()), abs2PSj, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat PQ, PQDj;
    std::cout << "hell" << std::endl;
    cv::mulSpectrums(makeComplex(Q_), P_, PQ, cv::DFT_COMPLEX_OUTPUT, conjB);    //conjP because we define P the other way around
    cv::mulSpectrums(PQ, D.at(j), PQDj, cv::DFT_COMPLEX_OUTPUT); 
    
    cv::Mat Q2 = Q_.mul(Q_);
    cv::Mat gl, pjgl;
    
    cv::mulSpectrums(PQDj-abs2PSj, makeComplex(1.0/Q2), gl, cv::DFT_COMPLEX_OUTPUT);
    
    shift(gl, gl, gl.cols/2, gl.rows/2);
    cv::idft(gl, gl, cv::DFT_REAL_OUTPUT);
    
    cv::multiply(gl, pj_pad, pjgl);
    
    cv::dft(pjgl, pjgl, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    shift(pjgl, pjgl, pjgl.cols/2, pjgl.rows/2);
    
    cv::Mat Pjpjgl;
    cv::mulSpectrums(pjgl, Pj_pad, Pjpjgl, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat grad = -2 * std::sqrt(J) * splitComplex(Pjpjgl).second;
                     
    for(unsigned int m = 0; m < M; ++m)
    {
      cv::Mat zBase = cv::Mat::zeros(D.front().size(), zernikeBase.at(m).type());
      zernikeBase.at(m).copyTo(selectCentralROI(zBase, zernikeBase.at(m).size()));
      
      if(!zernikeBase.at(m).empty())
      {
        g_.at<double>((j * M) + m, 0) = grad.dot(zBase);
      }
      else
      {
        g_.at<double>((j * M) + m, 0) = 0.0;
      }
    }
   
  }
 
  return g_;

}
