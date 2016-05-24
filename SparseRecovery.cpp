/*
 * SparseRecovery.cpp
 *
 *  Created on: Sep 01, 2015
 *      Author: dailos
 */
#include <limits>
#include <list>

#include "SparseRecovery.h"
#include "CustomException.h"
#include "ToolBox.h"
#include "FITS.h"

cv::Mat perform_projection(const cv::Mat& Phi0, const cv::Mat& y0)
{
  cv::Mat y, Phi;
  bool ComplexValued = complexToRealValued(Phi0, y0, Phi, y);

  //Methods: DECOMP_LU, DECOMP_CHOLESKY, DECOMP_SVD
  //Better implement as left division!!!! don't use this: cv::Mat x = Phi.inv(cv::DECOMP_SVD) * y;
  // Ax = b
  // x = cv::solve(A, b);     % A\b or mldivide(A,b)
  cv::Mat x;
  cv::solve(Phi, y, x, cv::DECOMP_NORMAL);

  
  if(ComplexValued)
  {
    cv::Mat xx_real = x( cv::Range(0, x.total()/2), cv::Range::all() ).clone();
    cv::Mat xx_imag = x( cv::Range(x.total()/2, x.total()), cv::Range::all() ).clone();
    std::vector<cv::Mat> x_v = {xx_real, xx_imag};
    cv::merge(x_v, x);
  }
  return x.clone(); 

}

cv::Mat perform_FISTA(const cv::Mat& Phi0, const cv::Mat& y0, const double& lambda)
{

  double EPSILON = 1e-16;
  unsigned int MAX_ITERS = 1600;
  //s: sparsity, non-zero elements expected in the solution
  cv::Mat y, Phi;
  bool ComplexValued = complexToRealValued(Phi0, y0, Phi, y);
  
  double sigsize = y.dot(y)/y.total();
  double old_err = sigsize;
  
  auto perform_soft_thresholding = [](const cv::Mat& x, const double& tau)-> cv::Mat
  {
    return cv::max( 0.0, 1.0 - tau/cv::max(cv::abs(x),1e-10) ).mul(x);
  };
  
  //operator callbacks
  auto F = [lambda](const cv::Mat& x)-> double {return lambda * cv::norm(x,cv::NORM_L1);};
  auto G = [Phi, y](const cv::Mat& x)-> double {return 0.5 * std::pow(cv::norm(y-Phi*x, cv::NORM_L2), 2.0);};
  
  //Proximal operator of F.
  auto ProxF = [lambda, perform_soft_thresholding](const cv::Mat& x, const double& tau) -> cv::Mat { return perform_soft_thresholding(x, lambda*tau); };
  
  //Gradient operator of G.
  auto GradG = [Phi, y](const cv::Mat& x) -> cv::Mat { return Phi.t() * (Phi*x-y); };
  
  //Lipschitz constant.
  double L = 1.0;
  //double L = std::pow(cv::norm(Phi, cv::NORM_L2), 2.0);

    
  //Main 'fista' algorithm
  double t = 1.0;
  cv::Mat x = cv::Mat::zeros(Phi.cols, 1, cv::DataType<double>::type);   //starting point
  cv::Mat yy = x.clone();
  cv::Mat xnew;
  double old_val = F(x)+G(x);
  double Lstep = 1.5;
  for (unsigned int iter = 0; iter < 2; ++iter)
  {
    
    //Backtracking: linesearch to find the best value for L
    for(unsigned int nline=0;nline<800;++nline)
    {
      cv::Mat GradGy = GradG(yy).clone();
      xnew = perform_soft_thresholding( yy - GradGy/L, lambda/L );
      cv::Mat stp = xnew-yy;
      double gxnew = G(xnew);
      if (gxnew <= G(yy) + stp.dot(GradGy) + (L/2.0) * stp.dot(stp) + F(xnew) )  break;
      else L = L * Lstep;
      //std::cout << "L: " << L << std::endl;
    }
    
    
    xnew = perform_soft_thresholding( yy - GradG(yy)/L, lambda/L );  //Or ProxF(yy - GradG(yy)/L, 1.0/L)
    double tnew = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
    yy = xnew + (t - 1.0) / (tnew)*(xnew-x);
    x = xnew.clone(); t = tnew;
    double new_val = F(x)+G(x);
    if(std::abs(old_val-new_val) < EPSILON){std::cout << "Solution found at iteration number " << iter << std::endl; break;}
    else old_val = new_val;
  }

  if(ComplexValued)
  {
    cv::Mat xx_real = x( cv::Range(0, x.total()/2), cv::Range::all() ).clone();
    cv::Mat xx_imag = x( cv::Range(x.total()/2, x.total()), cv::Range::all() ).clone();
    std::vector<cv::Mat> x_v = {xx_real, xx_imag};
    cv::merge(x_v, x);
  }
  return x.clone(); 
} 


cv::Mat perform_IHT(const cv::Mat& Phi0, const cv::Mat& y0, const unsigned int& s, const double& mu)
{
  double EPSILON = 1e-8;
  unsigned int MAX_ITERS = 800;
  
  //s: sparsity, non-zero elements expected in the solution
  cv::Mat y, Phi;
  bool ComplexValued = complexToRealValued(Phi0, y0, Phi, y);
  
  //Keep only k largest coefficients of 'p' and set to zero the rest
  auto perform_hard_thresholding = [](cv::Mat& p, const unsigned int& k)-> void
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

  
  double sigsize = y.dot(y)/y.total();
  double old_err = sigsize;
  
  cv::Mat Residual = y.clone();
  cv::Mat x = cv::Mat::zeros(Phi.cols, 1, cv::DataType<double>::type);   //Intial value for solution
  double MU( mu );
  bool zero_mu(MU == 0.0);   //means step-size should be computed at each iteration
  
  for(unsigned int iter =0; iter<MAX_ITERS; ++iter)
  {
    cv::Mat Px;
    if( zero_mu )
    {
      //Calculate optimal step size and do line search
      cv::Mat oldx = x.clone();
      cv::Mat oldPx = Phi * x;
      cv::Mat ind(x.size(), CV_8U, cv::Scalar(255));
      cv::Mat d = Phi.t() * Residual;
    
      if( cv::countNonZero(x) == 0)
      { // If the current vector is zero, we take the largest elements in d
        cv::Mat pp(cv::abs(d));
        for(unsigned int i=0;i<s;++i)
        {
          cv::Point maxLoc;
          cv::minMaxLoc(pp, nullptr,nullptr, nullptr, &maxLoc, ind);
          ind.at<char>(maxLoc.y, maxLoc.x) = 0;
        }
      }
      else
      {
        ind.setTo(0, x != 0);
      }
      
      cv::Mat id = d.clone();
      id.setTo(0.0, ind);
      cv::Mat Pd = Phi * id;
      MU = id.dot(id) / Pd.dot(Pd);
      
      x = oldx + MU * d;

      perform_hard_thresholding(x, s);
      Px = Phi * x;

      // Calculate step-size requirement 
      double val = cv::norm(x-oldx)/cv::norm(Px-oldPx);

      cv::Mat not_ind, xor_not_ind;
      cv::bitwise_not(ind, not_ind);
      // As long as the support changes and (MU > val*val), we decrease MU
      cv::bitwise_xor(not_ind, x!=0, xor_not_ind);

      while ( MU > 0.99*val*val && cv::countNonZero(xor_not_ind) != 0 && cv::countNonZero(not_ind) != 0 )
      {
        // We use a simple line search, halving MU in each step
        MU = MU/2.0;
        x = oldx + MU * d;
        perform_hard_thresholding(x, s);
        Px = Phi * x;
        // Calculate step-size requirement 
        val = cv::norm(x-oldx)/cv::norm(Px-oldPx);
      }
      //std::cout << "MU: " << MU << std::endl;
    }
    else
    {
      x = x + MU * (Phi.t() * Residual);
      // ####: x.setTo(0.0, cv::abs(x) < 0.0001);  //alternative approach that keeps only values with magnitude greater than lambda (e.g. 0.0001) in each iteration.
      perform_hard_thresholding(x, s);  //keeps exactly s largest elements in each iteration.
      Px = Phi * x;
    }
  
    Residual = y - Px;
    double err = Residual.dot(Residual) / Residual.total();
    if ( (old_err - err)/sigsize < EPSILON && iter >=2) { std::cout << "Solution found. At iteration number: " << iter << std::endl; break; }
    if ( err < EPSILON ) { std::cout << "Exact solution found. At iteration number: " << iter << std::endl; break; }
    old_err = err;
  }
  
  if(ComplexValued)
  {
    cv::Mat xx_real = x( cv::Range(0, x.total()/2), cv::Range::all() ).clone();
    cv::Mat xx_imag = x( cv::Range(x.total()/2, x.total()), cv::Range::all() ).clone();
    std::vector<cv::Mat> x_v = {xx_real, xx_imag};
    cv::merge(x_v, x);
  }

  return x.clone();
}

cv::Mat perform_BSBL(const cv::Mat& Phi0, const cv::Mat& y0, const NoiseLevel& LearnLambda, std::vector<double>& gamma_v, const unsigned int& blkLength)
{
  std::cout.precision( std::numeric_limits<double>::digits10 + 1);
  double EPSILON       =  1e-9;      // solution accurancy tolerance  
  unsigned int MAX_ITERS     = 3;        // maximum iterations

  cv::Mat y, Phi, Phi_0;
  bool ComplexValued = complexToRealValued(Phi0, y0, Phi, y);


  cv::Mat PhiPhi, sumPhiPhi, sqrtsumPhiPhi;
  cv::multiply(Phi, Phi, PhiPhi);
  cv::reduce(PhiPhi, sumPhiPhi, 0, CV_REDUCE_SUM);   //sum every column a reduce matrix to a single row
  sqrtsumPhiPhi = sqrtsumPhiPhi;
  cv::sqrt(sumPhiPhi, sqrtsumPhiPhi);
  cv::divide(Phi, cv::Mat::ones(Phi.rows,1,cv::DataType<double>::type) * sqrtsumPhiPhi, Phi);

  
  Phi.copyTo(Phi_0);   //Save a copy of the initial Phi for later use
  
  // scaling... 
  cv::Scalar mean, std;
  cv::meanStdDev(y, mean, std);
  double std_n = sqrt(std.val[0]*std.val[0]*y.total()/(y.total()-1));   //Std normalized to N-1 isntead of N
  if ((std_n < 0.4) || (std_n > 1)) y = 0.4*y/std_n;

  //stopping criteria used : (OldRMS-NewRMS)/RMS(x) < stopTol
  double sigsize = y.dot(y)/y.total();
  double old_err = sigsize;

  // Default Parameter Values for Any Cases
  bool PRINT         = true;          // don't show progress information
  bool intrablockCorrelation = false;          // adaptively estimate the covariance matrix B
  double PRUNE_GAMMA;
  double lambda;
  if(Phi.cols%blkLength != 0) throw CustomException("Invalid block length: Number of cols of Phi should be multiple of block lengh.");
  unsigned int blkNumber(Phi.cols/blkLength);    //total number of blocks
  
  //if (LearnLambda == NoiseLevel::Noiseless) { lambda = 1e-12;   PRUNE_GAMMA = 1e-3; }
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
    blkList.push_back( Block(k, k*blkLength, blkLength, gamma_v.at(k), cv::Mat::eye(blkLength, blkLength, cv::DataType<double>::type) ));
  }
  
  cv::Mat mu_x = cv::Mat::zeros(M, 1, cv::DataType<double>::type);
  unsigned int pos = 0;
  unsigned int count;
  size_t lsize = blkNumber;
  
  // Iteration: MAX_ITERS
  for (count = 1; count < MAX_ITERS; ++count)
  {
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
    //Matrix right division should be implemented cv::solve instead of by inverse multiplication
    // xA = b  => x = b/A but never use x = b * inv(A)!!!
    //They are mathematically equivalent but not the same when working with floating numbers
    //x = cv.solve( A.t(), b.t() ).t();  equivalent to b/A
    //to perform matrix right division: mrdivide(b,A)
    cv::Mat H;
    cv::solve(den.t(), Phi, H, cv::DECOMP_NORMAL);
    
    cv::Mat Hy = H.t() * y;
    cv::Mat HPhi = H.t() * Phi;
    
    
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
      invB = B.inv(cv::DECOMP_SVD); //Methods: DECOMP_LU, DECOMP_CHOLESKY, DECOMP_SVD
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
      
      
      blk->gamma( cv::trace(invB * blk->Cov_x()).val[0] / blk->Cov_x().cols );

//      //Alternative algorithm of computing gamma called BSBL_BO
//      //MATLAB code: gamma(i) = gamma_old(i)*norm( sqrtm(B{i})*Hy(currentSeg) )/sqrt(trace(HPhi(currentSeg,currentSeg)*B{i}));
//      cv::Mat eigenvalues, eigenvectors, sqrt_eigenvalues;
//      cv::eigen(B, eigenvalues, eigenvectors);
//      cv::sqrt(eigenvalues, sqrt_eigenvalues);
//      cv::Mat sqrtB = eigenvectors.t() * cv::Mat::diag(sqrt_eigenvalues) * eigenvectors;
//      double num = blk->gamma() * cv::norm( sqrtB * Hy(cv::Range(pos, pos + blk->length()), cv::Range::all()), cv::NORM_L2 );
//      double den = std::sqrt( cv::trace( HPhi(cv::Range(pos, pos + blk->length()), cv::Range(pos, pos + blk->length())) * B).val[0] );
//      blk->gamma( num / den );

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
  
  //test gamma
  gamma_v = std::vector<double>(blkNumber, 0.0);
  pos = 0;
  for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
  {
    gamma_v.at(blk->startLoc() / blk->length()) = blk->gamma();
  }
  //for_each(gamma_v.begin(), gamma_v.end(), [](const double& d){std::cout << d << std::endl;});
  ////
  
  
    // ================= Check stopping conditions, eyc. ==============
    if ( mu_x.total() == mu_old.total() )
    {
      //cv::absdiff(mu_old, mu_x, diff);
      cv::Mat diff = cv::abs(mu_old - mu_x);
      double maxVal;
      cv::minMaxLoc(diff, nullptr, &maxVal, nullptr, nullptr);
      if (maxVal < EPSILON)
      {
        std::cout << "Solution found." << std::endl;
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
  }
  
  
  // reconstruct the original signal &  Expand hyperparameyers
  cv::Mat x = cv::Mat::zeros(M,1, cv::DataType<double>::type);
  
  //for(Block blk : blkList)
  pos = 0;
  for(auto blk = blkList.begin(); blk != blkList.end(); ++blk)
  {
    mu_x(cv::Range(pos, pos + blk->length()), cv::Range::all()).copyTo( x(cv::Range(blk->startLoc(), blk->startLoc() + blk->length()), cv::Range::all()) );
    pos += blk->length();
  }
  
  cv::divide(x, sqrtsumPhiPhi.t(), x);
  std::cout << "x:" << x.t() << std::endl;
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
    std::cout << "gamma_est:" << std::endl;
    //for_each(gamma_est.begin(), gamma_est.end(), [](const double& d){std::cout << d << std::endl;});
  }
  
  if ((std_n < 0.4) | (std_n > 1))  x = x * std_n/0.4;
  return x.clone();
}


bool complexToRealValued(const cv::Mat& Phi0, const cv::Mat& y0, cv::Mat& Phi, cv::Mat& y)
{
  bool ComplexValued(false);
  if(y0.channels() == 2 || Phi0.channels() == 2)
  {
    //::complexToRealValued::
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
    
    ComplexValued = true;
  }
  else if(y0.channels() == 1 && Phi0.channels() == 1)
  {
    y0.copyTo(y);
    Phi0.copyTo(Phi);
    ComplexValued = false;
  }
  else throw CustomException("Invalid number of channels of input matrices.");
  
  return ComplexValued;
}
