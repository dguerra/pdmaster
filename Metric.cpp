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
#include "Zernikes.cpp"
#include "FITS.h"
#include "Fusion.h"
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
  unsigned int pupilSideLength = optimumSideLength(D.front().size().width/2, tsettings.pupilRadiousPixels());
  
  unsigned int K = D.size();   //number of images to use in the algorithm
  unsigned int M = zernikeBase.size();   //number of zernike coefficients to use in the representation of each image phase
  if(K * M != coeffs.total()) throw CustomException("Coeffcient vector should contain K*M elements.");
    
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());
  
  ////////Consider the case of two diversity images
  std::vector<double> diversityFactor = {0.0, -2.21209};
  cv::Mat z4 = Zernikes<double>::phaseMapZernike(4, pupilSideLength, tsettings.pupilRadiousPixels());
  double z4AtOrigen = Zernikes<double>::pointZernike(4, 0, 0);
  std::vector<cv::Mat> diversityPhase;
  for(double dfactor : diversityFactor)
  {
    //defocus zernike coefficient: c4 = dfactor * PI/(2.0*std::sqrt(3.0))
	  diversityPhase.push_back( (dfactor * 3.141592/(2.0 * std::sqrt(3.0))) * (z4 - z4AtOrigen));
  }
  ////////
  
  
  for(unsigned int k=0; k<K; ++k)
  {  //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
    cv::Mat pupilPhase_i = Zernikes<double>::phaseMapZernikeSum(pupilSideLength,tsettings.pupilRadiousPixels(), coeffs(cv::Range(k*M, k*M + M), cv::Range::all()));
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
    //in case of undersampling optimumSideLength is bigger then image size
    
    cv::accumulate(selectCentralROI(absSj2, D.front().size()), Q); //equivalent to Q += (absSj)^2 === absSj.mul(absSj);
  }
  
  TelescopeSettings tsettings(D.front().cols);
  Q.setTo(0, Zernikes<double>::phaseMapZernike(1, Q.cols, tsettings.cutoffPixel()) == 0);
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
    cv::mulSpectrums(Sj, selectCentralROI(dSj, Sj.size()), SjdSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj x dSj*
    cv::mulSpectrums(selectCentralROI(dSj, Sj.size()), Sj, dSjSj, cv::DFT_COMPLEX_OUTPUT, conjB);  //Sj* x dSj
    //in case of undersampling optimumSideLength is bigger then image size
    cv::Mat sumSjdSj = SjdSj + dSjSj;
    cv::accumulate(selectCentralROI(sumSjdSj, D.front().size()), dQ); //equivalent to Q += (absSj)^2 === absSj.mul(absSj);
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
    
    cv::mulSpectrums(D.at(j), selectCentralROI(Sj, D.front().size()), SjDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::accumulate(SjDj, P);   //equivalent to P += SjDj;
  }
  TelescopeSettings tsettings(D.front().cols);
  P.setTo(0, Zernikes<double>::phaseMapZernike(1, P.cols, tsettings.cutoffPixel()) == 0);
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
    
    cv::mulSpectrums(D.at(j), selectCentralROI(dSj, D.front().size()), dSjDj, cv::DFT_COMPLEX_OUTPUT, conjB);
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
  double L;
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

  /////Create L = sum{ abs(D0H - FHT0)^2 + abs(DkH - FHTk)^2 }
  cv::Mat accDiff = cv::Mat::zeros(F_.size(), F_.depth());
  double l1_norm(0.0);
  for(unsigned int j = 0; j < D.size(); ++j)
  {
    cv::Mat DjH;
    cv::mulSpectrums(D.at(j), makeComplex(H), DjH, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat FHSj;
    cv::Mat Sj = OS.at(j).otf().clone();    
    
    fftShift(Sj);   //shifts fft spectrums and takes energy to the center of the image
    TelescopeSettings tsettings(D.front().cols);
    Sj.setTo(0, Zernikes<double>::phaseMapZernike(1, Sj.cols, tsettings.cutoffPixel()) == 0);
    cv::mulSpectrums(F_, selectCentralROI(Sj, F_.size()), FHSj, cv::DFT_COMPLEX_OUTPUT);    
    cv::accumulateSquare(absComplex(DjH - FHSj), accDiff);
  }
  
  //Transfor obejct estimate F into curvelets coefficients
  L = cv::sum(accDiff).val[0];
  
  return L;
}
 

//Computes derivative of OTF with respect to an element of the zernike base
void Metric::compute_dSj(const OpticalSystem& osj, const cv::Mat& zernikeElement, cv::Mat& dSj)
{
  cv::Mat Pj = osj.generalizedPupilFunction();
  cv::Mat Pj_pad = cv::Mat::zeros(zernikeElement.size(), Pj.type());
  Pj.copyTo(selectCentralROI(Pj_pad, Pj.size()));
  cv::Mat ZH;
  cv::mulSpectrums(makeComplex(zernikeElement), Pj_pad, ZH, cv::DFT_COMPLEX_OUTPUT);
  
  cv::Mat cross;
  bool full(true), corr(true);
  convolveDFT(Pj_pad, ZH, cross, corr, full);
  cv::copyMakeBorder(cross, cross, 1, 0, 1, 0, cv::BORDER_CONSTANT);
  fftShift(cross);
  
  //cv::Mat cross = crosscorrelation(Pj_pad, ZH);
  cv::Mat H_ZH;
  
  cross.copyTo(H_ZH);
 
  cv::Mat H_ZHFlipped;
  cv::flip(H_ZH, H_ZHFlipped, -1); //flipCode => -1 < 0 means two axes flip
  shift(H_ZHFlipped, H_ZHFlipped, 1, 1);  //shift matrix => 1,1 means one pixel to the right, one pixel down
  cv::Mat diff = H_ZH - conjComplex(H_ZHFlipped);
  std::pair<cv::Mat, cv::Mat> splitComplexMatrix = splitComplex(diff);
  dSj = makeComplex((-1)*splitComplexMatrix.second, splitComplexMatrix.first).clone();//equivalent to multiply by imaginary unit i
  fftShift(dSj);
  
  //dSj = -1 * dSj;    //PLEASE LOOK INTO THIS
}


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
  cv::mulSpectrums(Q, Q, Q2, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(P, P, absP2, cv::DFT_COMPLEX_OUTPUT, conjB);
  noiseFilter(coeffs, D, zernikeBase, meanPowerNoise, P, Q, H);
  //Object estimate: F = (P/Q) x filter
  divSpectrums(P, Q, F, cv::DFT_COMPLEX_OUTPUT);
  cv::mulSpectrums(F, makeComplex(H), F, cv::DFT_COMPLEX_OUTPUT);   //Filter the object estimate out
  
  size_t J = OS.size();
  size_t M = zernikeBase.size();
  
  cv::Mat g = cv::Mat::zeros(J*M, 1, cv::DataType<std::complex<double> >::type);


  for(size_t j = 0; j < J; ++j)
  {
    cv::Mat Pj = OS.at(j).generalizedPupilFunction();
    
    cv::Mat Pj_pad = cv::Mat::zeros(D.front().size(), Pj.type());
    Pj.copyTo(selectCentralROI(Pj_pad, Pj.size()));
    
    //Why is not here normalization needed with generalized pupil function?!!!
    cv::Mat pj_pad = Pj_pad.clone();
    fftShift(pj_pad);
    
    cv::idft(pj_pad, pj_pad, cv::DFT_COMPLEX_OUTPUT);  //##ALERT: Usually it's cv::DFT_REAL_OUTPUT with idft
    cv::Mat gl, pjgl;
    
    cv::Mat FDj;
    cv::mulSpectrums(D.at(j),F, FDj, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::Mat Sj = OS.at(j).otf().clone();
    fftShift(Sj);
    TelescopeSettings tsettings(D.front().cols);
    Sj.setTo(0, Zernikes<double>::phaseMapZernike(1, Sj.cols, tsettings.cutoffPixel()) == 0);
    
    cv::Mat abs2F, abs2FSj;
    cv::mulSpectrums(F, F, abs2F, cv::DFT_COMPLEX_OUTPUT, conjB);
    cv::mulSpectrums(abs2F, selectCentralROI(Sj, abs2F.size()), abs2FSj, cv::DFT_COMPLEX_OUTPUT);
    gl = FDj - abs2FSj;
    
    //Put V value aside for a later use
    cv::Mat V;
    gl.copyTo(V);
    
    fftShift(gl);
    cv::idft(gl, gl, cv::DFT_COMPLEX_OUTPUT);    //##ALERT: Usually it's cv::DFT_REAL_OUTPUT with idft
    cv::mulSpectrums(makeComplex(splitComplex(gl).first), pj_pad, pjgl, cv::DFT_COMPLEX_OUTPUT);
    
    
    cv::dft(pjgl, pjgl, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    fftShift(pjgl);
    
    cv::Mat Pjpjgl;
    cv::mulSpectrums(pjgl, Pj_pad, Pjpjgl, cv::DFT_COMPLEX_OUTPUT, conjB);

    cv::Mat grad = -2 * splitComplex(Pjpjgl).second;

    for(size_t m = 0; m < M; ++m)
    {
    //Alternate way of calculate V
    cv::Mat PQ, PQD, absP2S, Va;
    cv::mulSpectrums(Q, P, PQ, cv::DFT_COMPLEX_OUTPUT, conjB);    
    cv::mulSpectrums(PQ, D.at(j), PQD, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(absP2, selectCentralROI(Sj, absP2.size()), absP2S, cv::DFT_COMPLEX_OUTPUT);
    divSpectrums(PQD-absP2S, Q2, Va, cv::DFT_COMPLEX_OUTPUT);
    //Va.copyTo(V);
      
      cv::Mat dSjV, dSj;
      compute_dSj(OS.at(j), zernikeBase.at(m), dSj);
      cv::mulSpectrums(selectCentralROI(dSj,V.size()), V, dSjV, cv::DFT_COMPLEX_OUTPUT, conjB);
      cv::mulSpectrums(V, selectCentralROI(dSj,V.size()), dSjV, cv::DFT_COMPLEX_OUTPUT, conjB);
      double gi_alt = cv::sum(dSjV).val[0];

      
      double gi = grad.dot(zernikeBase.at(m));
      //std::cout <<  "gi/gi_alt = " << gi << "/" << gi_alt << " = " << gi/gi_alt << std::endl;
      
      if(!zernikeBase.at(m).empty()) {g.at<std::complex<double> >((j * M) + m, 0) = std::complex<double>(gi ,gi);}
      else {g.at<std::complex<double> >((j * M) + m, 0) = std::complex<double>(0.0,0.0);}

      //if(!zernikeBase.at(m).empty()) {g.at<std::complex<double> >((j * M) + m, 0) = std::complex<double>(gi_alt ,gi_alt);}
      //else {g.at<std::complex<double> >((j * M) + m, 0) = std::complex<double>(0.0,0.0);}
    }
  }
  g = g / zernikeBase.front().total();
  
  
  //### Gradient of the regularization part: L1_NORM(Curvelets(Sj)).
  cv::Mat g_reg = cv::Mat::zeros(J*M, 1, cv::DataType<std::complex<double> >::type);
  

  return g + g_reg;
}

/*
if(false)
  {
    std::vector< std::vector<cv::Mat> > f_crvlts;   //Cruvelets coefficients of object estimate in measure space
    cv::Mat f(F);
    fftShift(f);
    cv::idft(f, f, cv::DFT_COMPLEX_OUTPUT);
    
    Curvelets::fdct(f, f_crvlts);   //will be used afterwards with -> sign(f_crvlts) 
    static int iii(0);iii++;
    //writeFITS(splitComplex(f).first, "../f"+std::to_string(iii)+".fits");
    
    
  for(size_t j = 0; j < J; ++j)
  {
    for(size_t m = 0; m < M; ++m)
    {
      //Compute dFdZm
      cv::Mat dP, dQ, QdP, PdQ, Q2, dF;
      compute_dP(D, zernikeBase.at(m), OS, dP);
      compute_dQ(D, zernikeBase.at(m), OS, dQ);
      cv::mulSpectrums(Q, dP, QdP, cv::DFT_COMPLEX_OUTPUT);
      cv::mulSpectrums(P, dQ, PdQ, cv::DFT_COMPLEX_OUTPUT);
      cv::mulSpectrums(Q, Q, Q2, cv::DFT_COMPLEX_OUTPUT);
      divSpectrums(QdP - PdQ, Q2, dF, cv::DFT_COMPLEX_OUTPUT);
      cv::Mat df(dF);
      fftShift(df);
      cv::idft(df, df, cv::DFT_COMPLEX_OUTPUT);
      
      std::vector< std::vector<cv::Mat> >  df_crvlts;
      Curvelets::fdct(df, df_crvlts);
      double low_subgradient(0.0), high_subgradient(0.0);
      
      for(unsigned int w = 0; w < df_crvlts.size(); ++w)
      { 
        for(auto s = 0; s < df_crvlts.at(w).size(); ++s)
        {
          cv::Mat signCrvlts, signCrvlts_;
          if( f_crvlts.at(w).at(s).total() != cv::countNonZero(f_crvlts.at(w).at(s))) std::cout << "Zero value curvelet coefficients" << std::endl;
          signCrvlts = cv::Mat::ones(f_crvlts.at(w).at(s).size(), f_crvlts.at(w).at(s).depth());
          signCrvlts_ = cv::Mat::ones(f_crvlts.at(w).at(s).size(), f_crvlts.at(w).at(s).depth());
          signCrvlts.setTo(-1, f_crvlts.at(w).at(s)<0.0);
          signCrvlts_.setTo(-1, f_crvlts.at(w).at(s)<=0.0);
          
          cv::Mat A, A_;
          cv::multiply(signCrvlts , df_crvlts.at(w).at(s), A);  //"sign" function considers zero as positive sign
          cv::multiply(signCrvlts_, df_crvlts.at(w).at(s), A_);  //"sign_" function considers zero as negative sign
          
          low_subgradient += cv::sum(cv::min(A, A_)).val[0];
          high_subgradient += cv::sum(cv::max(A, A_)).val[0];
        }
      }
      g_reg.at<std::complex<double> >((j * M) + m, 0) = std::complex<double>(low_subgradient, high_subgradient);
      if(low_subgradient != high_subgradient) std::cout << "Non smooth point!" << std::endl;
      
    }
  }
  }
  //In the end, every dimension has lower and upper bound subgradient (lower bound could be equal to upper bound).
*/
