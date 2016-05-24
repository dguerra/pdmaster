/*
 * WavefrontSensor.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#include "WavefrontSensor.h"
//#include "CustomException.h"
#include "BasisRepresentation.h"
//#include <cmath>
#include "ToolBox.h"
#include "OpticalSetup.h"
#include "FITS.h"
#include "Metric.h"
#include "ConvexOptimization.h"
#include "SparseRecovery.h"
#include <fstream>
//ANEXO
//How to get with python null space matrix from constraints, Q2
//import scipy
//import scipy.linalg
//A = scipy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
//Q, R = scipy.linalg.qr(A)

constexpr double PI = 3.14159265359;  //2 * acos(0.0);

//Other names, phaseRecovery, ObjectReconstruction, ObjectRecovery

WavefrontSensor::WavefrontSensor()
{
  diversityFactor_ = {0.0, -2.21209};
}

WavefrontSensor::~WavefrontSensor()
{
  // TODO Auto-generated destructor stub
}


cv::Mat
WavefrontSensor::WavefrontSensing(const std::vector<cv::Mat>& d, const std::vector<double>& meanPowerNoise)
{
  unsigned int numberOfZernikes = 20;   //total number of zernikes to be considered
  int M = numberOfZernikes;
  int K = d.size();
  cv::Mat Q2;
  partlyKnownDifferencesInPhaseConstraints(M, K, Q2);
  std::vector<cv::Mat> Q2_v = {Q2, cv::Mat::zeros(Q2.size(), Q2.type())};
  cv::Mat LEC;   //Linear equality constraints
  cv::merge(Q2_v, LEC);   //Build also the complex version of Q2

  //process each patch independently
  cv::Mat dd;
  std::vector<cv::Mat> d_w;
  std::vector<Metric> mtrc_v;
  std::vector<std::pair<cv::Range,cv::Range> > rngs;
  unsigned int pixelsBetweenTiles = (int)(d.front().cols);
  unsigned int tileSize = 34;
  OpticalSetup tsettings( tileSize );
  divideIntoTiles(d.front().size(), pixelsBetweenTiles, tileSize, rngs);
  //Random row selector: Pick incoherent measurements
  cv::Mat eye_nn = cv::Mat::eye(K*tileSize*tileSize, K*tileSize*tileSize, cv::DataType<double>::type);
  //unsigned int a = 228; //number of incoherent non-adaptive measurements
  unsigned int a = 400; // 42; //number of incoheren measurements
  cv::Mat shuffle_eye;
  shuffleRows(eye_nn, shuffle_eye);
  
  //Split 'a' into rngs.size() pieces
  
  std::vector<cv::Mat> A_v = {shuffle_eye(cv::Range(0, a), cv::Range::all()), cv::Mat::zeros(a, K*tileSize*tileSize, cv::DataType<double>::type)};
  cv::Mat A;
  cv::merge(A_v, A);
  
/*
  //Sensing matrix: Normalized random basis
  auto random_basis = []() -> void
  {
    cv::Mat Psi(M, N, cv::DataType<double>::type);
    cv::randn(Psi, cv::Scalar(1.0), cv::Scalar(1.0));
    cv::Mat Psi_norm;
    cv::multiply(Psi, Psi, Psi_norm);
    cv::reduce(Psi_norm, Psi_norm, 0, CV_REDUCE_SUM);   //sum every column a reduce matrix to a single row
    cv::sqrt(Psi_norm, Psi_norm);
    cv::divide(Psi, cv::Mat::ones(M,1,cv::DataType<double>::type) * Psi_norm, Psi);
  }
*/

  for(auto rng_i : rngs)
  {
    cv::Mat d_col;
    //get ready dataset format
    std::vector<cv::Mat> D;
    std::vector<cv::Mat> d_col_v;
    for(cv::Mat di : d)
    {
      cv::Mat Di;
      cv::dft(di(rng_i.first, rng_i.second), Di, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
      fftShift(Di);
      D.push_back(Di);
      cv::Mat Di_t(Di.t());
      d_col_v.push_back(Di_t.reshape(0, Di_t.total() ));
    }
    cv::vconcat(d_col_v, d_col);
    cv::gemm(A, d_col, 1.0, cv::Mat(), 1.0, d_col);  //Picks rows randomly
    
    d_w.push_back( d_col );
    mtrc_v.push_back( Metric(D, tsettings.pupilRadiousPixels(), numberOfZernikes) );
  }
  cv::vconcat(d_w, dd);
  
  //-----------------------BY MEANS OF CONVEX OPTIMIZATION:
  //Objective function and gradient of the objective function
  if(false)
  {
  for(auto mtrc : mtrc_v)
  {
    std::function<double(cv::Mat)> func = std::bind(&Metric::objective, &mtrc, std::placeholders::_1, meanPowerNoise);
    std::function<cv::Mat(cv::Mat)> dfunc = std::bind(&Metric::gradient, &mtrc, std::placeholders::_1, meanPowerNoise);
    ConvexOptimization minimizationKit;
    cv::Mat x0_conv = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);   //reset starting point
    
    //Lambda function that turn minimize function + constraints problem into minimize function lower dimension problem
    auto F_constrained = [] (cv::Mat x, std::function<double(cv::Mat)> func, const cv::Mat& Q2) -> double
    {
      return func(Q2*x);
    };
    
    auto DF_constrained = [] (cv::Mat x, std::function<cv::Mat(cv::Mat)> dfunc, const cv::Mat& Q2) -> cv::Mat
    {
      return Q2.t() * dfunc(Q2*x);
    };
      
    std::function<double(cv::Mat)> f_constrained = std::bind(F_constrained, std::placeholders::_1, func, Q2);
    std::function<cv::Mat(cv::Mat)> df_constrained = std::bind(DF_constrained, std::placeholders::_1, dfunc, Q2);
    //Define a new starting point with lower dimensions after reduction with contraints
    cv::Mat p_constrained = Q2.t() * x0_conv;
    ConvexOptimization min;
    min.perform_BFGS(p_constrained, f_constrained, df_constrained);
    x0_conv = Q2 * p_constrained;   //Go back to original dimensional 

    std::cout << "mimumum: " << x0_conv.t() << std::endl;
  }
  std::cout << "END OF CONVEX OPTIMIZATION"  << std::endl;
  }
  
  
  //-----------------------BY MEANS OF SPARSE RECOVERY:
  cv::Mat x0 = cv::Mat::zeros(rngs.size()*M*K, 1, cv::DataType<double>::type);   //x0: initial point
  std::vector<double> gamma_v(M*K, 1.0);
  for(unsigned int count=0;count<600;++count)
  {
    std::vector<cv::Mat> x0_vvv;
    cv::split(x0, x0_vvv);
    x0_vvv.at(0).copyTo(x0);
    
    cv::Mat_<std::complex<double> > blockMatrix_M;
    std::vector<cv::Mat> De_v;
    for(unsigned int t=0; t < rngs.size(); ++t)
    {
      cv::Mat jacob_i;
      mtrc_v.at(t).jacobian( x0(cv::Range(t*M*K, (t*M*K) + (M*K)), cv::Range::all()), meanPowerNoise, jacob_i );
      cv::gemm(A, jacob_i, 1.0, cv::Mat(), 1.0, jacob_i);   //Picks rows randomly
      cv::gemm(jacob_i, LEC, 1.0, cv::Mat(), 1.0, jacob_i);  //Apply constraints LECs
      cv::copyMakeBorder(blockMatrix_M, blockMatrix_M, 0, jacob_i.size().height, 0, jacob_i.size().width, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0) );
      cv::Rect rect(cv::Point(t*jacob_i.size().width, t*jacob_i.size().height), jacob_i.size() );
      jacob_i.copyTo(blockMatrix_M( rect ));
      cv::Mat De_i;
      mtrc_v.at(t).phi( x0(cv::Range(t*M*K, (t*M*K) + (M*K)), cv::Range::all()), meanPowerNoise, De_i );
      cv::gemm(A, De_i, 1.0, cv::Mat(), 1.0, De_i);   //Picks rows randomly
      De_v.push_back( De_i );
    }
    cv::Mat De;
    cv::vconcat(De_v, De);
    
    std::vector<cv::Mat> x0_v = {x0, cv::Mat::zeros(x0.size(), x0.type())};
    cv::merge(x0_v, x0);

    //Apply algorithm to get solution
    unsigned int blkLen = rngs.size();
    cv::Mat blockMatrix_M_r;
    reorderColumns(blockMatrix_M, M, blockMatrix_M_r);   //reorder columns so correlated data form a single block
    //std::cout << "blockMatrix_M.row(25): " << blockMatrix_M.row(25) << std::endl;
    //std::cout << "blockMatrix_M_r.row(25): " << blockMatrix_M_r.row(25) << std::endl;
    //LittleNoise, Noiseless, Noisy
    gamma_v = std::vector<double>(M*K, 1.0);
    cv::Mat coeffs = perform_BSBL(blockMatrix_M_r, dd - De, NoiseLevel::Noiseless, gamma_v, blkLen);
    //cv::Mat coeffs = perform_BSBL(blockMatrix_M_r, dd - De, NoiseLevel::LittleNoise, gamma_v, blkLen);
    //cv::Mat coeffs = perform_BSBL(blockMatrix_M_r, dd - De, NoiseLevel::Noisy, blkLen);
    //cv::Mat coeffs = perform_IHT (blockMatrix_M_r, dd - De, 4);
    //cv::Mat coeffs = perform_FISTA (blockMatrix_M_r, dd - De, 0.0000001);  //defautl:0.000001
    //cv::Mat coeffs = perform_projection(blockMatrix_M_r, dd - De);
    
    
    cv::Mat coeffs_r;
    reorderColumns(coeffs.t(), blockMatrix_M.cols/M, coeffs_r);
    
    cv::Mat coeffs_r_n(coeffs_r.t());
    
    //Undo constraints
    cv::Mat sol = cv::Mat::zeros(x0.size(), cv::DataType<std::complex<double> >::type);
    for(unsigned int t=0; t < rngs.size(); ++t)
    {
      cv::Mat sol_i;
      cv::gemm(LEC, coeffs_r_n(cv::Range(t*LEC.cols, (t*LEC.cols) + (LEC.cols)), cv::Range::all()), 1.0, cv::Mat(), 1.0, sol_i);
      sol_i.copyTo(sol(cv::Range(t*M*K, (t*M*K) + (M*K)), cv::Range::all()));
    }
    
    std::cout << "cv::norm(sol): " << cv::norm(sol) << std::endl;
    if(cv::norm(sol) < 1e-6 ) {std::cout << "Solution found" << std::endl; break;}
    x0 = x0 - sol;
    
    std::cout << "Solution number: " << count << std::endl;
    std::cout << "x0: " << x0.t() << std::endl;
  }
  
  return cv::Mat();  //mtrc.F();
}
