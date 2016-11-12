/*
 * Metric.cpp
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */
#include "Metric.h"
#include "ToolBox.h"
#include "OpticalSetup.h"
#include "CustomException.h"
#include "Zernike.h"
#include "FITS.h"

//nomenclature:
//Sj -> Optical Tranfer Function (OTF) for each independent optical path "j"
//D -> vector containing all data images for each optical path
//F -> Object estimate = P/Q


//Isolate meanPowerNoise into an independent noise filter class together with cutoff pixel information
//and cutoff_mask to apply to the data spectrums

//To bind the arguments of the member function "hello", within "object"
//Metric m;
//auto f = std::bind(&Metric::metric_from_coefs, &m, std::placeholders::_1);
Metric::Metric(const std::vector<cv::Mat>& D, const std::shared_ptr<Zernike>& zrnk, const double& meanPowerNoise)
  : zrnk_(zrnk), meanPowerNoise_(meanPowerNoise)
{
  for(auto Dk : D) D_.push_back(Dk.clone());
}


Metric::~Metric()
{
  // TODO Auto-generated destructor stub
}

void Metric::characterizeOpticalSystem(const cv::Mat& coeffs, std::vector<Optics>& OS)
{
  //check input dimensions
  if(coeffs.cols != 1) throw CustomException("Wrong input dimensions");
  
  //Start populating vector from scratch
  OS.clear();
  OpticalSetup tsettings(D_.front().size().width);
  unsigned int pupilSideLength = D_.front().size().width;
  
  unsigned int K = D_.size();   //number of images to use in the algorithm
  unsigned int M = zrnk_->base().size();   //number of zernike coefficients to use in the representation of each image phase
  if(K * M != coeffs.total()) throw CustomException("Coeffcient vector should contain K*M elements.");

  cv::Mat phase_div = cv::Mat::zeros(coeffs.size(), coeffs.type());
  phase_div.at<double>(M + 3, 0) = tsettings.k() * 3.141592/(2.0*std::sqrt(3.0));
  
  cv::Mat coeffs_new = coeffs + phase_div;

  for(unsigned int k=0; k<K; ++k)
  {  //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
    cv::Mat pupilPhase_i;
    zrnk_->synthesize(coeffs_new(cv::Range(k*M, k*M + M), cv::Range::all()), pupilPhase_i);
    OS.push_back(Optics(pupilPhase_i, zrnk_->base().at(0)));  //Characterized optical system
  }
}

void Metric::computeQ(const cv::Mat& coeffs, const std::vector<Optics>& OS, cv::Mat& Q)
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
  
  OpticalSetup tsettings(D_.front().cols);
  //Q.setTo(0, cutoff_mask_ == 0);
}

void Metric::compute_dQ(const cv::Mat& zernikeElement, const std::vector<Optics>& OS, const unsigned int& j, cv::Mat& dQ)
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


void Metric::computeP(const cv::Mat& coeffs, const std::vector<Optics>& OS, cv::Mat& P)
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
  OpticalSetup tsettings(D_.front().cols);
  //P.setTo(0, cutoff_mask_ == 0);
}

void Metric::compute_dP(const cv::Mat& zernikeElement, const std::vector<Optics>& OS, const unsigned int& j, cv::Mat& dP)
{
  //Simply compute one element of the sum in dP
  cv::Mat dSj;
  compute_dSj(OS.at(j), zernikeElement, dSj);
  bool conjB(true);
  cv::mulSpectrums(D_.at(j), dSj, dP, cv::DFT_COMPLEX_OUTPUT, conjB);
}


void Metric::noiseFilter(const cv::Mat& coeffs, const double& meanPowerNoise, const cv::Mat& P, const cv::Mat& Q, cv::Mat& filter)
{
  throw CustomException("Noise filter not implemented yet.");
  /*
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
  
  filter = 1.0 - (meanPowerNoise * (frac + fracFlip)/2.0);
  
  //remove peaks
  filter.setTo(0, filter < filter_lower_limit);
  filter.setTo(filter_upper_limit, filter > filter_upper_limit);
  
  
  //To zero-out frequencies beyond cutoff
  OpticalSetup tsettings(D_.front().cols);
  filter.setTo(0, cutoff_mask_ == 0);
  
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
  */
}


//objective function: L = sum_i{ |D_i - F * S_i|^2 }
//Objective function : L = sum_i{ |y - Φ(x)|^2 }
double Metric::objective( const cv::Mat& coeffs )
{
  double L(0.0);
  //Intermal metrics
  cv::Mat P, Q, H;
  std::vector<Optics> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, OS, Q);
  computeP(coeffs, OS, P);
  //noiseFilter(coeffs, meanPowerNoise_, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F_, cv::DFT_COMPLEX_OUTPUT);
  //cv::mulSpectrums(F_, makeComplex(H), F_, cv::DFT_COMPLEX_OUTPUT);

  size_t J = OS.size();
  
  /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 
  cv::Mat accDiff = cv::Mat::zeros(F_.size(), F_.depth());
  for(unsigned int j = 0; j < J; ++j)
  {
    cv::Mat DjH;
    //cv::mulSpectrums(D_.at(j), makeComplex(H), DjH, cv::DFT_COMPLEX_OUTPUT);
    DjH = D_.at(j).clone();
    cv::Mat FHSj;
    cv::Mat Sj = OS.at(j).otf().clone();
    
    fftShift(Sj);   //shifts fft spectrums and brings energy to the center of the image
    cv::mulSpectrums(F_, Sj, FHSj, cv::DFT_COMPLEX_OUTPUT);
    cv::accumulateSquare(absComplex(DjH - FHSj), accDiff);
  }
  
  L = cv::sum(accDiff).val[0];
  
/*
  //Alternative way
  cv::Mat absP2, absP2_Q;
  bool conjB(true);
  cv::mulSpectrums(P, P, absP2, cv::DFT_COMPLEX_OUTPUT, conjB);
  divSpectrums(absP2, Q, absP2_Q, cv::DFT_COMPLEX_OUTPUT);
  cv::Mat accD = cv::Mat::zeros(Q.size(), Q.type());
  for(auto Di : D_)
  {
    cv::Mat absDi2;
    cv::mulSpectrums(Di, Di, absDi2, cv::DFT_COMPLEX_OUTPUT, conjB);
    accD += absDi2;
  }
  std::cout << "Values for objective function: " << cv::sum(accD-absP2_Q).val[0] << " " << L << std::endl;
*/
  return L;
}

void Metric::phi( const cv::Mat& coeffs, cv::Mat& De )
{
  std::vector<cv::Mat> De_v;
  phi(coeffs, De_v);
  std::vector<cv::Mat> de_v;
  for(auto Dei : De_v )
  {
    cv::Mat Dei_t( Dei.t() );
    de_v.push_back(Dei_t.reshape(0, Dei_t.total() ));
  }
  cv::vconcat(de_v, De);
}

//Object estimate convoluted with OTFi for a given phase coefficient vector
void Metric::phi( const cv::Mat& coeffs, std::vector<cv::Mat>& De )
{
  cv::Mat P, Q, H, F;
  std::vector<Optics> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, OS, Q);
  computeP(coeffs, OS, P);
  //noiseFilter(coeffs, meanPowerNoise_, P, Q, H);

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

void Metric::jacobian( const cv::Mat& coeffs, cv::Mat& jacob )
{
  std::vector<std::vector<cv::Mat> > jacob_v;
  jacobian( coeffs, jacob_v );
  
  std::vector<cv::Mat> blockMatrix;
  std::vector<cv::Mat> res;
  for(size_t j = 0; j < jacob_v.size(); ++j)
  {
    std::vector<cv::Mat> blockMatrixRow;
    for(size_t m = 0; m < jacob_v.at(j).size(); ++m)
    {
      cv::Mat coo_t( jacob_v.at(j).at(m).t() );
      blockMatrixRow.push_back(coo_t.reshape(0, coo_t.total() ));
    }
        
    cv::Mat blockMatrixRow_M;
    cv::hconcat(blockMatrixRow, blockMatrixRow_M);
    blockMatrix.push_back(blockMatrixRow_M);
  }
  cv::Mat blockMatrix_M;
  cv::vconcat(blockMatrix, jacob);
}


//Compute the jacobian of Φ in the equation: y = Φ(x) + e
void Metric::jacobian( const cv::Mat& coeffs, std::vector<std::vector<cv::Mat> >& jacob )
{
  cv::Mat P, Q, H, F;
  std::vector<Optics> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeQ(coeffs, OS, Q);
  computeP(coeffs, OS, P);
  //noiseFilter(coeffs, meanPowerNoise_, P, Q, H);

  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  //cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);
  
  cv::Mat Q2;
  cv::mulSpectrums(Q, Q, Q2, cv::DFT_COMPLEX_OUTPUT);
  
  size_t J = OS.size();
  size_t M = zrnk_->base().size();
  
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
          compute_dSj(OS.at(j), zrnk_->base().at(m), dSj);
          cv::mulSpectrums(P, dSj, PdSj, cv::DFT_COMPLEX_OUTPUT);
        }
        compute_dP(zrnk_->base().at(m), OS, k, dP);
        cv::mulSpectrums(dP, Sj, dPSj, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(Q, dPSj+PdSj, lterm, cv::DFT_COMPLEX_OUTPUT);
        compute_dQ(zrnk_->base().at(m), OS, k, dQ);
        cv::mulSpectrums(dQ, PSj, rterm, cv::DFT_COMPLEX_OUTPUT);
        divSpectrums(lterm-rterm, Q2, tt, cv::DFT_COMPLEX_OUTPUT);
        vecM.push_back(tt);
      }
    }
   
    jacob.push_back(vecM);
  }
}


//Computes derivative of OTF with respect to an element of the zernike base
void Metric::compute_dSj(const Optics& osj, const cv::Mat& zernikeElement, cv::Mat& dSj)
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
cv::Mat Metric::gradient( const cv::Mat& coeffs )
{
  //Intermal metrics
  cv::Mat P, Q, H, F;
  std::vector<Optics> OS;
  characterizeOpticalSystem(coeffs, OS);
  computeP(coeffs, OS, P);
  computeQ(coeffs, OS, Q);
  //Some useful calculations
  bool conjB(true);
  //noiseFilter(coeffs, meanPowerNoise_, P, Q, H);
  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  //cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);   //Filter the object estimate out
  
  size_t J = OS.size();
  size_t M = zrnk_->base().size();
  
  cv::Mat g = cv::Mat::zeros(J*M, 1, cv::DataType<double>::type);

  Zernike zrnk;

  for(size_t j = 0; j < J; ++j)
  {
    cv::Mat FDj;
    cv::mulSpectrums(F, D_.at(j), FDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    OpticalSetup tsettings(D_.front().cols);
//    Sj.setTo(0, cutoff_mask_ == 0);   //restore when filter noise implemented again
    
    cv::Mat abs2F, abs2FSj;
    cv::mulSpectrums(F, F, abs2F, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::mulSpectrums(abs2F, Sj, abs2FSj, cv::DFT_COMPLEX_OUTPUT, conjB);
    
    //Put V value aside
    cv::Mat V = FDj - abs2FSj;
    
    for(size_t m = 0; m < M; ++m)
    {
      cv::Mat dSj, dSjV;
      compute_dSj(OS.at(j), zrnk_->base().at(m), dSj);
      cv::mulSpectrums(dSj, V, dSjV, cv::DFT_COMPLEX_OUTPUT);
      g.at<double>((j * M) + m, 0) =  cv::sum(dSjV).val[0];
    }
  }
  
/*
  //Alternate way of getting the gradient through dP and dQ
  std::vector<cv::Mat> De;
  std::vector<std::vector<cv::Mat> > jacob;
  phi(coeffs, De);
  jacobian(coeffs, jacob);
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
  
  std::cout << "g: " << g.t() << std::endl;
  std::cout << "g_phi: " << g_phi.t() << std::endl;
  std::cout << "g/g_phi: " << g.t()/g_phi.t() << std::endl;
  //g_phi.copyTo(g);
*/
  g = g / zrnk_->base().front().total();

  return g;
}
