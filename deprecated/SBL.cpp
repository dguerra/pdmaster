
/*
 * SBL.cpp
 *
 *  Created on: Nov 18, 2015
 *      Author: dailos
 */

#include "SBL.h"
#include "CustomException.h"
#include <algorithm>    // std::min_element, std::max_element
#include <limits>
#include <list>

cv::Mat BSBL::perform_BSBL(const cv::Mat& Phi0, const cv::Mat& y0, const NoiseLevel& LearnLambda, const unsigned int& blkLength)
{
  cv::Mat y, Phi, Phi_0;
  bool ComplexValued(false);
  if(y0.channels() == 2 || Phi0.channels() == 2)
  {
    complexToRealValued(Phi0, y0, Phi, y);
    ComplexValued = true;
  }
  else if(y0.channels() == 1 && Phi0.channels() == 1) 
  {
    y0.copyTo(y);
    Phi0.copyTo(Phi);
    ComplexValued = false;
  }
  else throw CustomException("Invalid number of channels of input matrices.");

  Phi.copyTo(Phi_0);   //Save a copy of the initial Phi for later use
 
  // scaling... 
  cv::Scalar mean, std;
  cv::meanStdDev(y, mean, std);
  double std_n = sqrt(std.val[0]*std.val[0]*y.total()/(y.total()-1));   //Std normalized to N-1 isntead of N
  if ((std_n < 0.4) | (std_n > 1)) y = 0.4*y/std_n;
  // Default Parameter Values for Any Cases
  double EPSILON       =  1e-3; // 1e-8;      // solution accurancy tolerance  
  unsigned int MAX_ITERS     = 600;        // maximum iterations
  bool PRINT         = true;          // don't show progress information
  bool intrablockCorrelation = true;          // adaptively estimate the covariance matrix B
  double PRUNE_GAMMA;
  double lambda;
  if(Phi.cols%blkLength != 0) throw CustomException("Invalid block length: Number of cols of Phi should be multiple of block lengh.");
  unsigned int blkNumber(Phi.cols/blkLength);    //total number of blocks
  
  if (LearnLambda == NoiseLevel::Noiseless) { lambda = 1e-12;   PRUNE_GAMMA = 1e-3; }
  else if(LearnLambda == NoiseLevel::LittleNoise) { lambda = 1e-3;    PRUNE_GAMMA = 1e-2; }
  else if(LearnLambda == NoiseLevel::Noisy) { lambda = 1e-3;    PRUNE_GAMMA = 1e-2; }
  else CustomException("Unrecognized Value for Input Argument LearnLambda");
  
  if(PRINT)
  {
    std::cout <<  "====================================================" << std::endl;
    std::cout <<  "           Running BSBL_EM......." << std::endl;
    std::cout <<  "           Information about parameters..." << std::endl;
    std::cout <<  "====================================================" << std::endl;
    std::cout <<  "PRUNE_GAMMA  : " << PRUNE_GAMMA << std::endl;
    std::cout <<  "lambda       : " << lambda << std::endl;
    //std::cout <<  "LearnLambda  : " << LearnLambda << std::endl;
    std::cout <<  "intrablockCorrelation    : " << intrablockCorrelation << std::endl;
    std::cout <<  "EPSILON      : " << EPSILON << std::endl;
    std::cout <<  "MAX_ITERS    : " << MAX_ITERS << std::endl;
  }
  
  // Initialization: [N,M] = size(Phi);
  unsigned int N = Phi.rows;
  unsigned int M = Phi.cols;
  std::cout << "N: " << N << ", M: " << M << std::endl;
  std::list<Block> blkList;
  
  //Initialize block list: same size blocks
  for(unsigned int k=0; k<blkNumber; ++k)
  {
    blkList.push_back( Block(k, k*blkLength, blkLength, 1.0, cv::Mat::eye(blkLength, blkLength, cv::DataType<double>::type) ));
  }
  
  cv::Mat mu_x = cv::Mat::zeros(M, 1, cv::DataType<double>::type);
  unsigned int count = 0;
  unsigned int pos = 0;
  size_t lsize = blkNumber;
  
  // Iteration
  while (1)
  {
    ++count;
    //std::cout << "SBL Iteration: " << count << std::endl;
    blkList.remove_if( [PRUNE_GAMMA](const Block& blk)->bool {return blk.gamma() < PRUNE_GAMMA;} );
    
    if(lsize != blkList.size())
    {
      lsize = blkList.size();
      if(blkList.empty())
      {
        std::cout <<  "====================================================================================" << std::endl;
        std::cout <<  "x becomes zero vector. The solution may be incorrect. " << std::endl;
        std::cout <<  "Current prune_gamma =" << PRUNE_GAMMA << ", and Current EPSILON = " << EPSILON << std::endl;
        std::cout <<  "Try smaller values of 'prune_gamma' and 'EPSILON' or normalize 'y' to unit norm." << std::endl;
        std::cout <<  "====================================================================================" << std::endl;
        break; 
      }
      // construct new Phi
      std::vector<cv::Mat> v_Phi;
      
      for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
      {
        //every image coeffcients are within the vector coeefs in the range (a,b), "a" inclusive, "b" exclusive
        cv::Mat ex = Phi_0(cv::Range::all(), cv::Range(blk->startLoc(), blk->startLoc() + blk->length()));
        v_Phi.push_back( ex.clone() );
      }
      cv::hconcat(v_Phi, Phi);
    }

    //=================== Compute new weights =================
    cv::Mat mu_old = mu_x.clone();  //Save solution vector for later
  
    cv::Mat PhiBPhi = cv::Mat::zeros(N, N, cv::DataType<double>::type);
    //for(Block blk : blkList)
    pos = 0;
    for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
    {
      cv::Mat snipPhi = Phi(cv::Range::all(), cv::Range(pos, pos + blk->length()));
      cv::accumulate(snipPhi * blk->Sigma_0() * snipPhi.t(), PhiBPhi);
      pos += blk->length();
    }
    
    //look the function cv::setIdentity
    cv::Mat lambdaI(PhiBPhi.size(), cv::DataType<double>::type);
    cv::setIdentity(lambdaI, lambda);
    
    cv::Mat den = PhiBPhi + lambdaI;
    cv::Mat H = Phi.t() * den.inv();
    cv::Mat Hy = H * y;
    cv::Mat HPhi = H * Phi;
    
    
    cv::Mat B(blkLength, blkLength, cv::DataType<double>::type); 
    cv::Mat invB(blkLength, blkLength, cv::DataType<double>::type);
    cv::Mat B0 = cv::Mat::zeros(blkLength, blkLength, cv::DataType<double>::type);
    
    std::vector<cv::Mat> v_mu_x;
//    for(Block blk : blkList)
    pos = 0;
    for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
    {
      cv::Mat mu_xi = blk->Sigma_0() * Hy(cv::Range(pos, pos + blk->length()), cv::Range::all());       // solution
      blk->Sigma_x( blk->Sigma_0() - blk->Sigma_0() * HPhi(cv::Range(pos, pos + blk->length()), cv::Range(pos, pos + blk->length())) * blk->Sigma_0() );
      blk->Cov_x( blk->Sigma_x() + mu_xi * mu_xi.t() );
      pos += blk->length();
      v_mu_x.push_back( mu_xi );
      // constrain all the blocks have the same correlation structure
      if(intrablockCorrelation == true)
      {
        B0 = B0 + blk->Cov_x()/blk->gamma();
      }
    }
    cv::vconcat( v_mu_x, mu_x );
    
    //=========== Learn correlation structure in blocks with Constraint 1 ===========
    // If blocks have the same size
    if (intrablockCorrelation == true)
    {
      // Constrain all the blocks have the same correlation structure (an effective strategy to avoid overfitting)
      double b = (cv::mean(B0.diag(1)) / cv::mean(B0.diag(0))).val[0];
      if (std::abs(b) >= 0.99) b = 0.98 * std::copysign(1.0, b);
      for(unsigned int j = 0; j < blkLength;++j) B.diag(j).setTo(std::pow(b, j));
      cv::completeSymm(B);   //Coppies the lower half into the upper half or the opposite
      invB = B.inv();
    }
    // do not consider correlation structure in each block
    else if(intrablockCorrelation == false)  //no consider intra-block correlation
    {
      B = cv::Mat::eye(blkLength, blkLength, cv::DataType<double>::type);
      invB = cv::Mat::eye(blkLength, blkLength, cv::DataType<double>::type);
    }
    
    //gamma_old = gamma;   #######################################################3
    double lambdaComp(0.0);
    
    //===========  estimate gamma(i) and lambda  =========== 
//    for(Block blk : blkList)
    pos = 0;
    for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
    {
      if(LearnLambda == NoiseLevel::Noisy)
      {
        lambdaComp += cv::trace( Phi(cv::Range::all(), cv::Range(pos, pos + blk->length())) * blk->Sigma_x() * 
                                 Phi(cv::Range::all(), cv::Range(pos, pos + blk->length())).t() ).val[0];
      }
      else if(LearnLambda == NoiseLevel::LittleNoise)
      {
        lambdaComp += cv::trace(blk->Sigma_x() * invB).val[0] / blk->gamma(); 
      }
      
      
      if(true)
      {
        blk->gamma( cv::trace(invB * blk->Cov_x()).val[0] / blk->Cov_x().cols );
      }
      else
      {
        //gamma(i) = gamma_old(i)*norm( sqrtm(B{i})*Hy(currentSeg) )/sqrt(trace(HPhi(currentSeg,currentSeg)*B{i}));
        cv::Mat eigenvalues, eigenvectors, sqrt_eigenvalues;
        cv::eigen(B, eigenvalues, eigenvectors);
        cv::sqrt(eigenvalues, sqrt_eigenvalues);
        cv::Mat sqrtB = eigenvectors.t() * cv::Mat::diag(sqrt_eigenvalues) * eigenvectors;
        double num = blk->gamma() * cv::norm( sqrtB * Hy(cv::Range(pos, pos + blk->length()), cv::Range::all()), cv::NORM_L2 );
        double den = std::sqrt( cv::trace( HPhi(cv::Range(pos, pos + blk->length()), cv::Range(pos, pos + blk->length())) * B).val[0] );
        blk->gamma( num / den );
      }
      
      blk->Sigma_0( B * blk->gamma() );
      
      pos += blk->length();
    }
    
    //LearnLambda == Noiseless, means no lambda value should be estimated
    if(LearnLambda == NoiseLevel::Noisy)
    {
      double normL2 = cv::norm(y - (Phi * mu_x), cv::NORM_L2);
      lambda = (normL2*normL2)/N + lambdaComp/N;
    }
    else if(LearnLambda == NoiseLevel::LittleNoise)
    {
      double normL2 = cv::norm(y - (Phi * mu_x), cv::NORM_L2);
      lambda = (normL2*normL2)/N + lambda * (mu_x.total() - lambdaComp)/N; 
    }
  
    // ================= Check stopping conditions, eyc. ==============
    if ( mu_x.total() == mu_old.total() )
    {
      ///////////////////////////////////////////////////////
      // reconstruct the original signal &  Expand hyperparameyers
      cv::Mat xx = cv::Mat::zeros(M,1, cv::DataType<double>::type);
      pos = 0;
      for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
      {
        mu_x(cv::Range(pos, pos + blk->length()), cv::Range::all()).copyTo( xx(cv::Range(blk->startLoc(), blk->startLoc() + blk->length()), cv::Range::all()) );
        pos += blk->length();
      }
      if(ComplexValued)
      {
        cv::Mat xx_real = xx( cv::Range(0, xx.total()/2), cv::Range::all() ).clone();
        cv::Mat xx_imag = xx( cv::Range(xx.total()/2, xx.total()), cv::Range::all() ).clone();
        std::vector<cv::Mat> xx_v = {xx_real, xx_imag};
        cv::merge(xx_v, xx);
      }
      //std::cout << "xx: " << xx.t() << std::endl;
      ///////////////////////////////////////////////////////
      
      
      cv::Mat diff;
      cv::absdiff(mu_old, mu_x, diff);
      double maxVal;
      //cv::minMaxLoc(diff, nullptr, &maxVal, nullptr, nullptr);
      maxVal = cv::norm(diff);
      if (maxVal < EPSILON)
      {
        std::cout << "B: " << B << std::endl;
        std::cout << "lambda: " << lambda << std::endl;
        break;
      }
    }
    if (PRINT) 
    {
      //std::cout << " iters: " << count << std::endl <<
      //             " num coeffs: " << blkList.size() << std::endl;
//                   " min gamma: " << min(gamma) << std::endl <<
//                   " gamma change: " << max(abs(gamma - gamma_old)) << std::endl <<
//                   " mu change: " << dmu << std::endl;
    }
    if (count >= 300)  //MAX_ITERS
    {
      if(PRINT)
      {
        std::cout << "Reach max iterations. Stop." << std::endl;
      }
      break;
    }
  }
  
  
  // reconstruct the original signal &  Expand hyperparameyers
  cv::Mat x = cv::Mat::zeros(M,1, cv::DataType<double>::type);
  std::vector<double> gamma_est(blkNumber, 0.0);
  
  //for(Block blk : blkList)
  pos = 0;
  for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
  {
    mu_x(cv::Range(pos, pos + blk->length()), cv::Range::all()).copyTo( x(cv::Range(blk->startLoc(), blk->startLoc() + blk->length()), cv::Range::all()) );
    pos += blk->length();
  }
  if(ComplexValued)
  {
    cv::Mat xx_real = x( cv::Range(0, M/2), cv::Range::all() ).clone();
    cv::Mat xx_imag = x( cv::Range(M/2, M), cv::Range::all() ).clone();
    std::vector<cv::Mat> x_v = {xx_real, xx_imag};
    cv::merge(x_v, x);
  }
  std::cout << "x: " << x.t() << std::endl;
  
  if(PRINT)
  {
    std::cout << "Number of iterations: " << count << std::endl;
    //std::cout << "gamma_est:" << std::endl;
    //for_each(gamma_est.begin(), gamma_est.end(), [](const double& d){std::cout << d << std::endl;});
  }
  
  if ((std_n < 0.4) | (std_n > 1))  x = x * std_n/0.4;
  return x.clone();
}

void BSBL::complexToRealValued(const cv::Mat& Phi0, const cv::Mat& y0, cv::Mat& Phi, cv::Mat& y)
{
  if(y0.channels() == 2)
  {
    std::vector<cv::Mat> y0_chnnls;
    cv::split(y0, y0_chnnls);
    cv::vconcat(y0_chnnls, y);
  }
  else
  {
    std::vector<cv::Mat> y0_chnnls = {y0, cv::Mat::zeros(y0.size(), y0.type())};
    cv::vconcat(y0_chnnls, y);
  }
  
  if(Phi0.channels() == 2)
  {
    std::vector<cv::Mat> Phi0_chnnls;
    cv::split(Phi0, Phi0_chnnls);
    std::vector<cv::Mat> upperMatrix = {Phi0_chnnls.at(0), -1.0 * Phi0_chnnls.at(1)};
    std::vector<cv::Mat> lowerMatrix = {Phi0_chnnls.at(1),        Phi0_chnnls.at(0)};
    cv::Mat uM, lM;
    cv::hconcat(upperMatrix, uM);
    cv::hconcat(lowerMatrix, lM);
    std::vector<cv::Mat> PhiM = {uM, lM};
    cv::vconcat(PhiM, Phi);
  }
  else
  {
    std::vector<cv::Mat> upperMatrix = {Phi0, cv::Mat::zeros(Phi0.size(), Phi0.type())};
    std::vector<cv::Mat> lowerMatrix = {cv::Mat::zeros(Phi0.size(), Phi0.type()), Phi0};
    cv::Mat uM, lM;
    cv::hconcat(upperMatrix, uM);
    cv::hconcat(lowerMatrix, lM);
    std::vector<cv::Mat> PhiM = {uM, lM};
    cv::vconcat(PhiM, Phi);
  }
}