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
#include "Curvelets.h"

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
  
  //Start populating vector from scratch
  OS.clear();
  TelescopeSettings tsettings(D.front().size().width);
  unsigned int pupilSideLength = D.front().size().width;
  
  unsigned int K = D.size();   //number of images to use in the algorithm
  unsigned int M = zernikeBase.size();   //number of zernike coefficients to use in the representation of each image phase
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

void Metric::computeQ(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                      const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& Q)
{
   //We use 'depth' here instead of 'type' because we want 1-channel image
  Q = cv::Mat::zeros(D.front().size(), D.front().type());
  for(size_t j = 0; j < D.size(); ++j)
  { 
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    cv::Mat absSj2;
    bool conjB(true);
    cv::mulSpectrums(Sj, Sj, absSj2, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj x Sj* == |Sj|^2    
    cv::accumulate(absSj2, Q); //equivalent to Q += (absSj)^2 === absSj.mul(absSj);
  }
  
  TelescopeSettings tsettings(D.front().cols);
  Q.setTo(0, Zernikes::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::compute_dQ(const std::vector<cv::Mat>& D, const cv::Mat& zernikeElement, 
                        const std::vector<OpticalSystem>& OS, cv::Mat& dQ)
{
   //We use 'depth' here instead of 'type' because we want 1-channel image
  dQ = cv::Mat::zeros(D.front().size(), D.front().type());
  for(size_t j = 0; j < D.size(); ++j)
  { 
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    cv::Mat dSj, SjdSj, dSjSj;
    compute_dSj(OS.at(j), zernikeElement, dSj);
    
    bool conjB(true);
    cv::mulSpectrums(Sj, dSj, SjdSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj x dSj*
    cv::mulSpectrums(dSj, Sj, dSjSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj* x dSj
    //in case of undersampling optimumSideLength is bigger then image size
    cv::Mat sumSjdSj = SjdSj + dSjSj;
    cv::accumulate(sumSjdSj, dQ); //equivalent to Q += (absSj)^2 === absSj.mul(absSj);
  }
}


void Metric::computeP(const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                            const std::vector<double>& meanPowerNoise, const std::vector<OpticalSystem>& OS, cv::Mat& P)
{
  P = cv::Mat::zeros(D.front().size(), D.front().type());

  //P = accumulate over J { Dj * conj(Sj) } 
  for(size_t j = 0; j < D.size(); ++j)
  {
    cv::Mat SjDj;
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    bool conjB(true);
    
    cv::mulSpectrums(D.at(j), Sj, SjDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::accumulate(SjDj, P);   //equivalent to P += SjDj;
  }
  TelescopeSettings tsettings(D.front().cols);
  P.setTo(0, Zernikes::phaseMapZernike(1, P.cols, tsettings.cutoffPixel()) == 0);
}

void Metric::compute_dP(const std::vector<cv::Mat>& D, const cv::Mat& zernikeElement, 
                        const std::vector<OpticalSystem>& OS, cv::Mat& dP)
{
  dP = cv::Mat::zeros(D.front().size(), D.front().type());

  //dP = accumulate over J { Dj * derivative(conj(Sj)) } 
  for(size_t j = 0; j < D.size(); ++j)
  {
    cv::Mat dSj, dSjDj;
    compute_dSj(OS.at(j), zernikeElement, dSj);
    bool conjB(true);
    
    cv::mulSpectrums(D.at(j), dSj, dSjDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::accumulate(dSjDj, dP);   //equivalent to dP += dSjDj;
  }
}


void Metric::noiseFilter(const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                 const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter)
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
  TelescopeSettings tsettings(D.front().cols);
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
double Metric::objectiveFunction( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                                  const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise )
{
  double L(0.0);
  //Intermal metrics
  cv::Mat P, Q, H;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS, Q);
  computeP(coeffs, D, zernikeBase, meanPowerNoise, OS, P);
  noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F_, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F_, makeComplex(H), F_, cv::DFT_COMPLEX_OUTPUT);

  size_t J = OS.size();
  
  /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 }
  cv::Mat accDiff = cv::Mat::zeros(F_.size(), F_.depth());
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat DjH;
    cv::mulSpectrums(D.at(j), makeComplex(H), DjH, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat FHSj;
    cv::Mat Sj = OS.at(j).otf().clone();
    
    fftShift(Sj);   //shifts fft spectrums and takes energy to the center of the image
    cv::mulSpectrums(F_, Sj, FHSj, cv::DFT_COMPLEX_OUTPUT);
    cv::accumulateSquare(absComplex(DjH - FHSj), accDiff);
  }
  
  L = cv::sum(accDiff).val[0];
  
  return L;
}

//Object estimate convoluted with OTFi for a given phase coefficient vector
void Metric::phi( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                                  const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, std::vector<cv::Mat>& De )
{
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS, Q);
  computeP(coeffs, D, zernikeBase, meanPowerNoise, OS, P);
  noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);
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

void Metric::compute_dphi( const cv::Mat& coeffs, const std::vector<cv::Mat>& D,
                           const std::vector<cv::Mat>& zernikeBase, const std::vector<double>& meanPowerNoise, std::vector<std::vector<cv::Mat> >& jacob )
{
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS, Q);
  computeP(coeffs, D, zernikeBase, meanPowerNoise, OS, P);
  noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);
  
  cv::Mat Q2;
  cv::mulSpectrums(Q, Q, Q2, cv::DFT_COMPLEX_OUTPUT);
  
  size_t J = OS.size();
  size_t M = zernikeBase.size();
  bool conjB(true);
  
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    cv::Mat PSj;
    cv::mulSpectrums(P, Sj, PSj, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> vecM;
    for(size_t m = 0; m < M; ++m)
    {
      cv::Mat accDdSk = cv::Mat::zeros(Sj.size(), Sj.type());
      cv::Mat accSdSk = cv::Mat::zeros(Sj.size(), Sj.type());
      for(unsigned int k = 0; k < J; ++k)
      {
        cv::Mat DdSk, SdSk, dSk;
        compute_dSj(OS.at(k), zernikeBase.at(m), dSk);
        cv::mulSpectrums(D.at(k), dSk, DdSk, cv::DFT_COMPLEX_OUTPUT, conjB);
        cv::accumulate(DdSk, accDdSk);
        
        cv::Mat Sk = OS.at(k).otf().clone();
        fftShift(Sk);
        cv::mulSpectrums(Sk, dSk, SdSk, cv::DFT_COMPLEX_OUTPUT, conjB);
        cv::accumulate(SdSk, accSdSk);
      }
      cv::Mat dSj, PdSj;
      compute_dSj(OS.at(j), zernikeBase.at(m), dSj);
      cv::mulSpectrums(P, dSj, PdSj, cv::DFT_COMPLEX_OUTPUT);

      cv::Mat accDdSkSj, lterm, rterm;
      cv::mulSpectrums(accDdSk, Sj, accDdSkSj, cv::DFT_COMPLEX_OUTPUT);
      cv::mulSpectrums(Q, accDdSkSj + PdSj, lterm, cv::DFT_COMPLEX_OUTPUT);
      cv::mulSpectrums(PSj, accSdSk + conjComplex(accSdSk), rterm, cv::DFT_COMPLEX_OUTPUT);
      cv::Mat tt;
      divSpectrums(lterm-rterm, Q2, tt, cv::DFT_COMPLEX_OUTPUT);
      vecM.push_back(tt);
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
cv::Mat Metric::gradient( const cv::Mat& coeffs, const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& zernikeBase, 
                          const std::vector<double>& meanPowerNoise )
{
  //Intermal metrics
  cv::Mat P, Q, H, F;
  std::vector<OpticalSystem> OS;
  characterizeOpticalSystem(coeffs, D, zernikeBase, OS);
  computeP(coeffs, D, zernikeBase, meanPowerNoise, OS, P);
  computeQ(coeffs, D, zernikeBase, meanPowerNoise, OS, Q);
  //Some useful calculations
  cv::Mat absP2, Q2;
  bool conjB(true);
  noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, P, Q, H);
  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);   //Filter the object estimate out
  
  size_t J = OS.size();
  size_t M = zernikeBase.size();
  
  cv::Mat g = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);


  for(size_t j = 0; j < J; ++j)
  { 
    cv::Mat FDj;
    cv::mulSpectrums(F, D.at(j), FDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    TelescopeSettings tsettings(D.front().cols);
    Sj.setTo(0, Zernikes::phaseMapZernike(1, Sj.cols, tsettings.cutoffPixel()) == 0);
    
    cv::Mat abs2F, abs2FSj;
    cv::mulSpectrums(F, F, abs2F, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::mulSpectrums(abs2F, Sj, abs2FSj, cv::DFT_COMPLEX_OUTPUT, conjB);
    
    //Put V value aside
    cv::Mat V = FDj - abs2FSj;
    
    for(size_t m = 0; m < M; ++m)
    {      
      cv::Mat dSj, dSjV;
      compute_dSj(OS.at(j), zernikeBase.at(m), dSj);
      cv::mulSpectrums(dSj, V, dSjV, cv::DFT_COMPLEX_OUTPUT);
      double gi = cv::sum(dSjV).val[0];  //Is it possible to do it with cv::dot??
      if(!zernikeBase.at(m).empty()) {g.at<double>((j * M) + m, 0) = gi;}
      else {g.at<double>((j * M) + m, 0) = 0.0;}
    }
  }
  
  //Alternate way of getting the gradient through jacobian of phi
  cv::Mat g_phi = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);
  std::vector<std::vector<cv::Mat> > jacob;
  std::vector<cv::Mat> De;
  compute_dphi(coeffs, D, zernikeBase, meanPowerNoise, jacob);
  phi(coeffs, D, zernikeBase, meanPowerNoise, De);
  for(size_t j = 0; j < J; ++j)
  { 
    for(size_t m = 0; m < M; ++m)
    {
      cv::Mat lterm, rterm, diff;
      cv::mulSpectrums(jacob.at(j).at(m), De.at(j), lterm, cv::DFT_COMPLEX_OUTPUT, conjB);
      cv::mulSpectrums(D.at(j), De.at(j), rterm, cv::DFT_COMPLEX_OUTPUT, conjB);
      diff = lterm - rterm;
      g_phi.at<double>((j * M) + m, 0) = cv::sum(diff).val[0];
    }
  }
  std::cout << "g: " << g.t() << std::endl;
  std::cout << "g_phi: " << g_phi.t() << std::endl;
  cv::Mat gg = g/g_phi;
  std::cout << "g/g_phi: " << gg.t() << std::endl;
  
  g = g / zernikeBase.front().total();

  return g;
}
