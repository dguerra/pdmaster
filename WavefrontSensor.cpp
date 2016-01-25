/*
 * WavefrontSensor.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#include "WavefrontSensor.h"
//#include "CustomException.h"
#include "Zernikes.h"
//#include <cmath>
#include "PDTools.h"
#include "TelescopeSettings.h"
#include "FITS.h"
#include "Metric.h"
#include "Minimization.h"
#include "CompressedSensing.h"
#include "SBL.h"
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
  cv::Size d_size = d.front().size();
  for(cv::Mat di : d)
  {
    if (d_size != di.size())
    {
      std::cout << "Input dataset images must be iqual size" << std::endl;
      //throw CustomException("Input dataset images must be iqual size");
    }
  }
  
  TelescopeSettings tsettings(d_size.width);
  unsigned int numberOfZernikes = 14;   //total number of zernikes to be considered

  cv::Mat d_col;
  std::vector<cv::Mat> d_col_v;
  std::vector<cv::Mat> D;
  for(cv::Mat di : d)
  {
    cv::Mat Di;
    cv::dft(di, Di, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    fftShift(Di);
    D.push_back(Di);
    cv::Mat Di_t(Di.t());
    d_col_v.push_back(Di_t.reshape(0, Di_t.total() ));
  }
  cv::vconcat(d_col_v, d_col);

  unsigned int pupilSideLength = optimumSideLength(d_size.width/2, tsettings.pupilRadiousPixels());
  std::cout << "pupilRadiousPixels: " << tsettings.pupilRadiousPixels() << std::endl;
  
  //double pupilRadiousP = tsettings.pupilRadiousPixels();
  cv::Mat pupilAmplitude = Zernikes::phaseMapZernike(1, pupilSideLength, tsettings.pupilRadiousPixels());

  Metric mtrc(D, tsettings.pupilRadiousPixels());
  //Objective function and gradient of the objective function
  std::function<double(cv::Mat)> func = std::bind(&Metric::objective, &mtrc, std::placeholders::_1, meanPowerNoise);
  std::function<cv::Mat(cv::Mat)> dfunc = std::bind(&Metric::gradient, &mtrc, std::placeholders::_1, meanPowerNoise);
  
  int M = numberOfZernikes;
  int K = D.size();
  cv::Mat Q2;
  partlyKnownDifferencesInPhaseConstraints(M, K, Q2);
  
  //Used this: http://davidstutz.de/matrix-decompositions/matrix-decompositions/householder
  //double q2_oneCoeff[] = {0, 0, 0, -0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70710678118655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  //Q2 = cv::Mat(K*M, 1, cv::DataType<double>::type, q2_oneCoeff);
  
  //p: initial point; Q2: null space of constraints; objFunction: function to be minimized; gradFunction: first derivative of objFunction 
  cv::Mat p = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
  
  if(true)
  {
    ////////////////////////////////////////////////
    //Verify jacobian of Φ
    //double data[] = { 0.0, 0.0, 0.0, 0.0, 0.21, 0.22, 0.0, 0.0, 0.0, 0.0, 0.75, 0.74, 0.0, 0.0,
                      //0.0, 0.0, 0.0, 0.0, 0.21, 0.22, 0.0, 0.0, 0.0, 0.0, 0.75, 0.74, 0.0, 0.0 };
    cv::Mat x0 = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
    //cv::Mat x0(M*K, 1, cv::DataType<double>::type, data);
    std::vector<std::vector<cv::Mat> > jacob;

    for(;;)
    {
      std::vector<cv::Mat> x0_vvv;
      cv::split(x0, x0_vvv);
      x0_vvv.at(0).copyTo(x0);
      
      mtrc.jacobian( x0, meanPowerNoise, jacob );
      //mtrc.jacobian( 0.1 * cv::Mat::ones(M*K, 1, cv::DataType<double>::type), meanPowerNoise, jacob );
      
      std::vector<cv::Mat> De;
      mtrc.phi( x0, meanPowerNoise, De );
      
      std::vector<cv::Mat> de_v;
      cv::Mat de;
      for(auto Dei : De )
      {
        cv::Mat Dei_t( Dei.t() );
        de_v.push_back(Dei_t.reshape(0, Dei_t.total() ));
      }
      cv::vconcat(de_v, de);
      
      std::vector<cv::Mat> blockMatrix;
      std::vector<cv::Mat> res;
      for(size_t j = 0; j < jacob.size(); ++j)
      {
        std::vector<cv::Mat> blockMatrixRow;
        for(size_t m = 0; m < jacob.at(j).size(); ++m)
        {
          cv::Mat coo_t( jacob.at(j).at(m).t() );
          blockMatrixRow.push_back(coo_t.reshape(0, coo_t.total() ));
        }
        
        cv::Mat blockMatrixRow_M;
        cv::hconcat(blockMatrixRow, blockMatrixRow_M);
        blockMatrix.push_back(blockMatrixRow_M);
      }
      cv::Mat blockMatrix_M;
      cv::vconcat(blockMatrix, blockMatrix_M);
      ////////////////////////////////////////////////

      //Pick incoherent measurements
      auto shuffleRows = [](const cv::Mat &matrix) -> cv::Mat
      {
        std::vector <int> seeds;
        for (int cont = 0; cont < matrix.rows; cont++)
        {
          seeds.push_back(cont);
        }
        cv::theRNG() = cv::RNG( cv::getTickCount() );
        cv::randShuffle(seeds);
      
        cv::Mat output;
        for (int cont = 0; cont < matrix.rows; cont++)
        {
          output.push_back(matrix.row(seeds[cont]));
        }
        return output;
      };
      cv::Mat eye_nn = cv::Mat::eye(blockMatrix_M.rows, blockMatrix_M.rows, cv::DataType<double>::type);
      unsigned int a = 64; //blockMatrix_M.rows/2; //800;  //number of incoheren measurements
      cv::Mat shuffle_eye = shuffleRows(eye_nn);
      std::vector<cv::Mat> A_v = {shuffle_eye(cv::Range(0, a), cv::Range::all()), cv::Mat::zeros(a, blockMatrix_M.rows, cv::DataType<double>::type)};
      cv::Mat A;
      cv::merge(A_v, A);
      cv::Mat new_phi = cv::Mat::zeros(a, blockMatrix_M.cols, cv::DataType<std::complex<double> >::type);
      cv::gemm(A, blockMatrix_M, 1.0, new_phi, 1.0, new_phi);
      new_phi.copyTo(blockMatrix_M);
    
      
      std::vector<cv::Mat> x0_v = {x0, cv::Mat::zeros(x0.size(), x0.type())};
      cv::merge(x0_v, x0);
      /*
      //Simulate measurement
      std::vector<cv::Mat> y_v = {cv::Mat::zeros(blockMatrix_M.rows, 1, x0.type()), cv::Mat::zeros(blockMatrix_M.rows, 1, x0.type())};
      cv::Mat y;
      cv::merge(y_v, y);
      cv::gemm(blockMatrix_M, x0, 1.0, y, 1.0, y);
      */
      
      //Apply constraints
      std::vector<cv::Mat> Q2_v = {Q2, cv::Mat::zeros(Q2.size(), Q2.type())};
      cv::Mat Q2_cmplx;
      cv::merge(Q2_v, Q2_cmplx);
      cv::Mat con = cv::Mat::zeros(blockMatrix_M.rows, Q2_cmplx.cols, Q2_cmplx.type());
      cv::gemm(blockMatrix_M, Q2_cmplx, 1.0, con, 1.0, con);
      
      //Apply algorithm to get solution
      unsigned int blkLen = 2;
      cv::Mat new_y = cv::Mat::zeros(a, 1, cv::DataType<std::complex<double> >::type);
      cv::gemm(A, d_col - de, 1.0, new_y, 1.0, new_y);
      double norm_l2 = 1.0; //cv::norm(new_y, cv::NORM_L2);
      cv::Mat coeffs = BSBL::perform_BSBL(con/norm_l2, new_y/norm_l2, BSBL::NoiseLevel::Noiseless, blkLen);
      //cv::Mat coeffs = BSBL::perform_BSBL(con, y, BSBL::NoiseLevel::Noiseless, blkLen);
      
      
      //Undo constraints
      cv::Mat sol = cv::Mat::zeros(Q2_cmplx.rows, 1, Q2_cmplx.type());
      cv::gemm(Q2_cmplx, coeffs, 1.0, sol, 1.0, sol);
      x0 = x0-sol;
      //x0 = sol.clone();
      std::cout << "x0: " << x0.t() << std::endl;
    }
  }
  
  
  
  Minimization minimizationKit;
  
  minimizationKit.minimize(p, Q2, func, dfunc, Minimization::Method::FISTA);
  std::cout << "mimumum: " << p.t() << std::endl;
  return mtrc.F();
}


/*
  //First attempt to solve through IHT
  auto vec = [](const std::vector<cv::Mat> &matrixV, cv::Mat& vector)
  {
    std::vector<cv::Mat> vv;
    for(size_t i = 0; i < matrixV.size(); ++i)
    {
      for(unsigned int j = 0; j < matrixV.at(i).cols; ++j)
      {
        vv.push_back(matrixV.at(i).col(j)); 
      }
    }
    cv::vconcat(vv, vector);
  };
  
  //Keep only k largest  coefficients
  auto hardThreshold = [](cv::Mat& p, const unsigned int& k)
  {
    cv::Mat mask = cv::Mat::ones(p.size(), CV_8U);
    cv::Mat pp(cv::abs(p));
    for(unsigned int i=0;i<k;++i)
    {
      cv::Point maxLoc;
      cv::minMaxLoc(pp, nullptr,nullptr, nullptr, &maxLoc, mask);
      mask.at<char>(maxLoc.y, maxLoc.x) = 0;
    }
    p.setTo(0.0, mask);
  };
  
  cv::Mat x0 = cv::Mat::zeros(K*M, 1, cv::DataType<double>::type);
  cv::Mat x0_imag = cv::Mat::zeros(K*M, 1, cv::DataType<double>::type);
  double mu(1.0);
  unsigned int sparsity(4);
  for(unsigned int j =0; j<500; ++j)
  {
    std::vector<cv::Mat> De;
    cv::Mat jacobian, measurement, observation;
    mtrc.phi(x0, D, zBase, meanPowerNoise, De);
    mtrc.compute_dphi(x0, D, zBase, meanPowerNoise, jacobian);
    vec(De, measurement);
    vec(D, observation);
    cv::Mat est = mu * (observation - measurement);
    cv::Mat zer = cv::Mat::zeros(est.size(), est.type());
    std::vector<cv::Mat> x0_ = {x0, x0_imag};
    cv::Mat x0_c;
    cv::merge(x0_, x0_c);
    cv::gemm(conjComplex(jacobian), est, -1.0, x0_c, 1.0, x0_c, cv::GEMM_1_T);
    x0 = splitComplex(x0_c).first;
    hardThreshold(x0, sparsity);
    
    std::cout << "x0: " << x0.t() << std::endl;
  }
*/