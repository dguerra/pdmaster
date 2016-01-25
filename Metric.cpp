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
#include "Zernikes.h"

//Isolate meanPowerNoise into an independent noise filter function within Metric class
//Build zernike base only once and keep it inside Metric class as a private member

//To bind the arguments of the member function "hello", within "object"
//Metric m;
//auto f = std::bind(&Metric::metric_from_coefs, &m, std::placeholders::_1);
Metric::Metric(const std::vector<cv::Mat>& D, const double& pupilRadiousPxls)
{
  pupilRadiousPxls_ = pupilRadiousPxls;
  
  for(auto Dk : D) D_.push_back(Dk.clone());
  
}

Metric::~Metric()
{
  // TODO Auto-generated destructor stub
}

void Metric::characterizeOpticalSystem(const cv::Mat& coeffs, std::vector<OpticalSystem>& OS)
{
  if( zernikeBase_.empty() == true )
  {
    zernikeBase_ = Zernikes::zernikeBase(coeffs.total()/D_.size(), D_.front().size().width, pupilRadiousPxls_);
  }
  
  //check input dimensions
  if(coeffs.cols != 1) throw CustomException("Wrong input dimensions");
  
  //Start populating vector from scratch
  OS.clear();
  TelescopeSettings tsettings(D_.front().size().width);
  unsigned int pupilSideLength = D_.front().size().width;
  
  unsigned int K = D_.size();   //number of images to use in the algorithm
  unsigned int M = zernikeBase_.size();   //number of zernike coefficients to use in the representation of each image phase
  if(K * M != coeffs.total()) throw CustomException("Coeffcient vector should contain K*M elements.");
    
  cv::Mat pupilAmplitude = Zernikes::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());

  const double lambda = 4.69600e-7;
  const double pi = 3.141592;
  
  ////////Consider the case of two diversity images
  std::vector<double> diversityFactor = {0.0, tsettings.k()};
  cv::Mat z4 = Zernikes::phaseMapZernike(4, pupilSideLength, tsettings.pupilRadiousPixels() );
  double z4AtOrigen = z4.at<double>(z4.cols/2, z4.rows/2);
  std::vector<cv::Mat> diversityPhase;
  for(double dfactor : diversityFactor)
  {
    //defocus zernike coefficient: c4 = dfactor * PI/(2.0*std::sqrt(3.0))
	  diversityPhase.push_back( (dfactor * 3.141592/(2.0*std::sqrt(3.0))) * (z4 - z4AtOrigen));
    //diversityPhase.push_back(dfactor * z4);
  }
  ////////
  
  for(unsigned int k=0; k<K; ++k)
  {  //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
    cv::Mat pupilPhase_i = Zernikes::phaseMapZernikeSum(pupilSideLength,tsettings.pupilRadiousPixels(), coeffs(cv::Range(k*M, k*M + M), cv::Range::all()));
    OS.push_back(OpticalSystem(pupilPhase_i + diversityPhase.at(k), pupilAmplitude));  //Characterized optical system
  }
 
}

void Metric::computeQ(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& Q)
{
   //We use 'depth' here instead of 'type' because we want 1-channel image
  Q = cv::Mat::zeros(D_.front().size(), D_.front().type());
  for(size_t j = 0; j < D_.size(); ++j)
  { 
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    cv::Mat absSj2;
    bool conjB(true);
    cv::mulSpectrums(Sj, Sj, absSj2, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj x Sj* == |Sj|^2    
    cv::accumulate(absSj2, Q); //equivalent to Q += (absSj)^2 === absSj.mul(absSj);
  }
  
  TelescopeSettings tsettings(D_.front().cols);
  Q.setTo(0, Zernikes::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::compute_dQ(const cv::Mat& zernikeElement, const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dQ)
{
  //We use 'depth' here instead of 'type' because we want 1-channel image
  cv::Mat Sj = OS.at(j).otf().clone();
  fftShift(Sj);
  cv::Mat dSj, SjdSj, dSjSj;
  compute_dSj(OS.at(j), zernikeElement, dSj);
    
  bool conjB(true);
  cv::mulSpectrums(Sj, dSj, SjdSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj x dSj*
  cv::mulSpectrums(dSj, Sj, dSjSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj* x dSj
  dQ = SjdSj + dSjSj;
}


void Metric::computeP(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& P)
{
  P = cv::Mat::zeros(D_.front().size(), D_.front().type());

  //P = accumulate over J { Dj * conj(Sj) } 
  for(size_t j = 0; j < D_.size(); ++j)
  {
    cv::Mat SjDj;
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    bool conjB(true);
    
    cv::mulSpectrums(D_.at(j), Sj, SjDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::accumulate(SjDj, P);   //equivalent to P += SjDj;
  }
  TelescopeSettings tsettings(D_.front().cols);
  P.setTo(0, Zernikes::phaseMapZernike(1, P.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::compute_dP(const cv::Mat& zernikeElement, const std::vector<OpticalSystem>& OS, const unsigned int& j, cv::Mat& dP)
{
  //Simply compute one element of the sum in dP
  cv::Mat dSj;
  compute_dSj(OS.at(j), zernikeElement, dSj);
  bool conjB(true);
  cv::mulSpectrums(D_.at(j), dSj, dP, cv::DFT_COMPLEX_OUTPUT, conjB);
}


void Metric::noiseFilter(const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter)
{
  const double filter_upper_limit(1.0);
  const double filter_lower_limit(0.1);
 
  cv::Mat absP = absComplex(P);
  cv::Mat frac;
  cv::divide(splitComplex(Q).first, absP.mul(absP), frac);  //Both Q and absP are single channel images, real matrices
  
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
  TelescopeSettings tsettings(D_.front().cols);
  filter.setTo(0, Zernikes::phaseMapZernike(1, filter.cols, tsettings.cutoffPixel()) == 0);
  
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


//objective function: L = sum_i{ |D_i - F * S_i|^2 }
//Objective function : L = sum_i{ |y - Φ(x)|^2 }
double Metric::objective( const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise )
{
  if( zernikeBase_.empty() == true )
  {
    zernikeBase_ = Zernikes::zernikeBase(coeffs.total()/D_.size(), D_.front().size().width, pupilRadiousPxls_);
  }
  
  double L(0.0);
  //Intermal metrics
  cv::Mat P, Q, H;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, meanPowerNoise, OS, Q);
  computeP(coeffs, meanPowerNoise, OS, P);
  noiseFilter(coeffs, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F_, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F_, makeComplex(H), F_, cv::DFT_COMPLEX_OUTPUT);

  size_t J = OS.size();
  
  /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 
  cv::Mat accDiff = cv::Mat::zeros(F_.size(), F_.depth());
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat DjH;
    cv::mulSpectrums(D_.at(j), makeComplex(H), DjH, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat FHSj;
    cv::Mat Sj = OS.at(j).otf().clone();
    
    fftShift(Sj);   //shifts fft spectrums and takes energy to the center of the image
    cv::mulSpectrums(F_, Sj, FHSj, cv::DFT_COMPLEX_OUTPUT);
    cv::accumulateSquare(absComplex(DjH - FHSj), accDiff);
  }
  
  L = cv::sum(accDiff).val[0];
/*
  cv::Mat absP2, absP2_Q;
  bool conjB;
  cv::mulSpectrums(P, P, absP2, cv::DFT_COMPLEX_OUTPUT, conjB);
  divSpectrums(absP2, Q, absP2_Q, cv::DFT_COMPLEX_OUTPUT);
  cv::Mat accD = cv::Mat::zeros(Q.size(), Q.type());
  for(auto Di : D)
  {
    cv::Mat absDi2;
    cv::mulSpectrums(Di, Di, absDi2, cv::DFT_COMPLEX_OUTPUT, conjB);
    accD += absDi2;
  }
  std::cout << cv::sum(accD-absP2_Q).val[0] << " " << L << std::endl;
  */
  return L;
}

//Object estimate convoluted with OTFi for a given phase coefficient vector
void Metric::phi( const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, std::vector<cv::Mat>& De )
{
  if( zernikeBase_.empty() == true )
  {
    zernikeBase_ = Zernikes::zernikeBase(coeffs.total()/D_.size(), D_.front().size().width, pupilRadiousPxls_);
  }
  
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, meanPowerNoise, OS, Q);
  computeP(coeffs, meanPowerNoise, OS, P);
  noiseFilter(coeffs, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  //cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);
  De.clear();
  
  size_t J = OS.size();
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat FSj;
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);   //shifts fft spectrums and takes energy to the center of the image
    cv::mulSpectrums(F, Sj, FSj, cv::DFT_COMPLEX_OUTPUT);
    De.push_back(FSj);
  }
}

//Compute the jacobian of Φ in the equation: y = Φ(x) + e
void Metric::jacobian( const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise, std::vector<std::vector<cv::Mat> >& jacob )
{
  if( zernikeBase_.empty() == true )
  {
    zernikeBase_ = Zernikes::zernikeBase(coeffs.total()/D_.size(), D_.front().size().width, pupilRadiousPxls_);
  }
  
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, meanPowerNoise, OS, Q);
  computeP(coeffs, meanPowerNoise, OS, P);
  noiseFilter(coeffs, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  //cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);
  
  cv::Mat Q2;
  cv::mulSpectrums(Q, Q, Q2, cv::DFT_COMPLEX_OUTPUT);
  
  size_t J = OS.size();
  size_t M = zernikeBase_.size();
  
  jacob.clear();
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    cv::Mat PSj;
    cv::mulSpectrums(P, Sj, PSj, cv::DFT_COMPLEX_OUTPUT);
    
    std::vector<cv::Mat> vecM;
    for(unsigned int k = 0; k < J; ++k)
    {
      for(size_t m = 0; m < M; ++m)
      {
        cv::Mat PdSj = cv::Mat::zeros(Sj.size(), Sj.type());
        cv::Mat dPSj, dSj, dP, dQ, lterm, rterm, tt;
        if(j == k)
        {
          compute_dSj(OS.at(j), zernikeBase_.at(m), dSj);
          cv::mulSpectrums(P, dSj, PdSj, cv::DFT_COMPLEX_OUTPUT);
        }
        compute_dP(zernikeBase_.at(m), OS, k, dP);
        cv::mulSpectrums(dP, Sj, dPSj, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(Q, dPSj+PdSj, lterm, cv::DFT_COMPLEX_OUTPUT);
        compute_dQ(zernikeBase_.at(m), OS, k, dQ);
        cv::mulSpectrums(dQ, PSj, rterm, cv::DFT_COMPLEX_OUTPUT);
        divSpectrums(lterm-rterm, Q2, tt, cv::DFT_COMPLEX_OUTPUT);
        vecM.push_back(tt);
      }
    }
   
    jacob.push_back(vecM);
  }
}


//Computes derivative of OTF with respect to an element of the zernike base
void Metric::compute_dSj(const OpticalSystem& osj, const cv::Mat& zernikeElement, cv::Mat& dSj)
{
  cv::Mat Pj = osj.generalizedPupilFunction();
  cv::Mat ZH;
  cv::mulSpectrums(makeComplex(zernikeElement), Pj, ZH, cv::DFT_COMPLEX_OUTPUT);
  
  cv::Mat cross;
  bool full(false), corr(true);
  convolveDFT(Pj, ZH, cross, corr, full);
  fftShift(cross);
  cv::Mat H_ZH;
  cross.copyTo(H_ZH);
  cv::Mat H_ZHFlipped;
  cv::flip(H_ZH, H_ZHFlipped, -1); //flipCode => -1 < 0 means two axes flip
  shift(H_ZHFlipped, H_ZHFlipped, 1, 1);  //shift matrix => 1,1 means one pixel to the right, one pixel down
  cv::Mat diff = H_ZH - conjComplex(H_ZHFlipped);
  
  std::pair<cv::Mat, cv::Mat> splitComplexMatrix = splitComplex(diff);
  dSj = makeComplex((-1)*splitComplexMatrix.second, splitComplexMatrix.first).clone();//equivalent to multiply by imaginary unit i
  fftShift(dSj);
}


//gradient of the whole objective function: L = sum_i{ |D_i - F * S_i|^2 } with respect to every parameter of the phase
cv::Mat Metric::gradient( const cv::Mat& coeffs, const std::vector<double>& meanPowerNoise )
{
  if( zernikeBase_.empty() == true )
  {
    zernikeBase_ = Zernikes::zernikeBase(coeffs.total()/D_.size(), D_.front().size().width, pupilRadiousPxls_);
  }
  
  //Intermal metrics
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeP(coeffs, meanPowerNoise, OS, P);
  computeQ(coeffs, meanPowerNoise, OS, Q);
  //Some useful calculations
  bool conjB(true);
  noiseFilter(coeffs, meanPowerNoise, P, Q, H);
  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);   //Filter the object estimate out
  
  size_t J = OS.size();
  size_t M = zernikeBase_.size();
  
  cv::Mat g = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);


  for(size_t j = 0; j < J; ++j)
  { 
    cv::Mat FDj;
    cv::mulSpectrums(F, D_.at(j), FDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    TelescopeSettings tsettings(D_.front().cols);
    Sj.setTo(0, Zernikes::phaseMapZernike(1, Sj.cols, tsettings.cutoffPixel()) == 0);
    
    cv::Mat abs2F, abs2FSj;
    cv::mulSpectrums(F, F, abs2F, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::mulSpectrums(abs2F, Sj, abs2FSj, cv::DFT_COMPLEX_OUTPUT, conjB);
    
    //Put V value aside
    cv::Mat V = FDj - abs2FSj;
    
    for(size_t m = 0; m < M; ++m)
    {      
      cv::Mat dSj, dSjV;
      compute_dSj(OS.at(j), zernikeBase_.at(m), dSj);
      cv::mulSpectrums(dSj, V, dSjV, cv::DFT_COMPLEX_OUTPUT);
      if(!zernikeBase_.at(m).empty()) {g.at<double>((j * M) + m, 0) =  cv::sum(dSjV).val[0];}  //Is it possible to do it with cv::dot??
      else {g.at<double>((j * M) + m, 0) = 0.0;}
    }
  }
  
  
  //Alternate way of getting the gradient through dP and dQ
  std::vector<cv::Mat> De;
  std::vector<std::vector<cv::Mat> > jacob;
  phi(coeffs, meanPowerNoise, De);
  jacobian(coeffs, meanPowerNoise, jacob);
  cv::Mat g_phi = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);
  
  for(size_t j = 0; j < J; ++j)
  { 
    cv::Mat diff = D_.at(j)-De.at(j);
    for(size_t m = 0; m < J*M; ++m)
    { 
      cv::Mat tt;
      //cv::mulSpectrums(diff, jacob.at(k).at(m), tt, cv::DFT_COMPLEX_OUTPUT, conjB);
      cv::mulSpectrums(diff, jacob.at(j).at(m), tt, cv::DFT_COMPLEX_OUTPUT, conjB);
      g_phi.at<double>(m, 0) += cv::sum(tt).val[0];
    }
  }
  
  //std::cout << "g: " << g.t() << std::endl;
  //std::cout << "g_phi: " << g_phi.t() << std::endl;
  //g_phi.copyTo(g);
  
  g = g / zernikeBase_.front().total();

  return g;
}
