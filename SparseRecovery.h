/*
 * SparseRecovery.h
 *
 *  Created on: Sep 01, 2015
 *      Author: dailos
 */

#ifndef SPARSERECOVERY_H_
#define SPARSERECOVERY_H_

#include <iostream>
#include "opencv2/opencv.hpp"
//y = ϕ(x) where x is the sparse vector and y the data

enum class NoiseLevel {Noiseless, LittleNoise, Noisy};
cv::Mat perform_projection(const cv::Mat& Phi0, const cv::Mat& y0);
cv::Mat perform_FISTA(const cv::Mat& Phi0, const cv::Mat& y0, const double& lambda);
cv::Mat perform_SBL(const cv::Mat& Phi0, const cv::Mat& y0, const NoiseLevel& LearnLambda, std::vector<double>& gamma_v);
cv::Mat perform_BSBL(const cv::Mat& Phi0, const cv::Mat& y0, const NoiseLevel& LearnLambda, std::vector<double>& gamma_v, const unsigned int& blkLength = 1);
cv::Mat perform_IHT(const cv::Mat& Phi0, const cv::Mat& y0, const unsigned int& s, const double& mu = 0.0);
bool complexToRealValued(const cv::Mat& Phi0, const cv::Mat& y0, cv::Mat& Phi, cv::Mat& y);

class Block
{
  public:
    Block(const unsigned int& num, const unsigned int& loc, const unsigned int& len, const double& gam, const cv::Mat& Sig) : number_(num), startLoc_(loc), length_(len), gamma_(gam), Sigma_0_(Sig.clone()){};
    virtual ~Block(){};       // Destructor
    unsigned int startLoc() const {return startLoc_;};
    unsigned int length()   const {return length_;};
    unsigned int number()   const {return number_;};
    
    //Getters
    double gamma()    const {return gamma_;};
    cv::Mat Sigma_0() const {return Sigma_0_;};
    cv::Mat Sigma_x() const {return Sigma_x_;};
    cv::Mat Cov_x()   const {return Cov_x_;};
    //Setters
    void gamma  (const double&   gmm) {gamma_ = gmm;};
    void Sigma_0(const cv::Mat& sgm0) {sgm0.copyTo(Sigma_0_);};
    void Sigma_x(const cv::Mat& sgmx) {sgmx.copyTo(Sigma_x_);};
    void Cov_x  (const cv::Mat& covx) {covx.copyTo(Cov_x_);};
    
  private:
    unsigned int number_;
    unsigned int startLoc_;
    unsigned int length_;
    double gamma_;
    cv::Mat Sigma_0_;
    cv::Mat Sigma_x_;
    cv::Mat Cov_x_;
};

#endif /* SPARSERECOVERY_H_ */


// % BSBL-EM: Recover block sparse signal (1D) exploiting intra-block correlation, given the block partition.
// %          It is the Cluster-SBL (Type I) in the ICASSP 2012 paper
// %          (Reference [1])
// %
// %          The algorithm solves the inverse problem for the block sparse
// %          model with known block partition:
// %                        y = Phi * x + v
// %
// %
// % ============================== INPUTS ============================== 
// %   Phi         : N X M known matrix
// %
// %   y           : N X 1 measurement vector 
// %
// %   blkListtartLoc : Start location of each block
// %   
// %   LearnLambda : (1) If LearnLambda = 1, use the lambda learning rule for generaly noisy 
// %                     cases (SNR<=20dB) (thus the input lambda is just as initial value)
// %                 (2) If LearnLambda = 2, use the lambda learning rule for high SNR cases (SNR>20dB)
// %                 (3) If LearnLambda = 0, do not use the lambda learning rule, but use the input 
// %                     lambda value as its final value.
// %                 
// %
// % [varargin values -- in most cases you can use the default values]
// %
// %  'LEARNTYPE'    : LEARNTYPE = 0: Ignore intra-block correlation
// %                   LEARNTYPE = 1: Exploit intra-block correlation 
// %                 [ Default: LEARNTYPE = 1 ]
// %
// %  'PRUNE_GAMMA'  : threshold to prune out small gamma_i 
// %                   (generally, 10^{-3} or 10^{-2})
// %
// %  'LAMBDA'       : user-input value for lambda
// %                  [ Default: LAMBDA=1e-14 when LearnLambda=0; LAMBDA=std(y)*1e-2 in noisy cases ]
// %
// %  'MAX_ITERS'    : Maximum number of iterations.
// %                 [ Default value: MAX_ITERS = 800 ]
// %
// %  'EPSILON'      : Solution accurancy tolerance parameter 
// %                 [ Default value: EPSILON = 1e-8   ]
// %
// %  'PRINT'        : Display flag. If = 1: show output; If = 0: supress output
// %                 [ Default value: PRINT = 0        ]
// %
// % ==============================  OUTPUTS ============================== 
// %   Result : A structured data with:
// %      Result.x          : the estimated block sparse signal
// %      Result.gamma_used : indexes of nonzero groups in the sparse signal
// %      Result.gamma_est  : the gamma values of all the groups of the signal
// %      Result.B          : the final value of the B
// %      Result.count      : iteration times
// %      Result.lambda     : the final value of lambda
// %
// %
// % ========================= Command examples  =============================
// %   < Often-used command >
// %    For general noisy environment:
// %           learnlambda = 1;
// %           Result =  BSBL_EM(Phi, y, blkListtartLoc, learnlambda);  
// %
// %    For high SNR cases (SNR >= 25 dB):
// %           learnlambda = 2;
// %           Result =  BSBL_EM(Phi, y, blkListtartLoc, learnlambda);   
// %
// %    For noiseless cases:
// %           learnlambda = 0;
// %           Result =  BSBL_EM(Phi, y, blkListtartLoc, learnlambda);  
// %
// %    < Full-Command Example >
// %           Result =  BSBL_EM(Phi, y, blkListtartLoc, learnlambda, ...
// %                                                 'LEARNTYPE', 1,...
// %                                                 'lambda', 1e-3,...
// %                                                 'prune_gamma',1e-2,...
// %                                                 'MAX_ITERS', 800,...
// %                                                 'EPSILON', 1e-8,...
// %                                                 'PRINT',0);
// %
// % ================================= See Also =============================
// %   EBSBL_EM,   BSBL_BO,  EBSBL_BO,  BSBL_L1,  EBSBL_L1,  TMSBL,    TSBL      
// %
// % ================================ Reference =============================
// %   [1] Zhilin Zhang, Bhaskar D. Rao, Recovery of Block Sparse Signals 
// %       Using the Framework of Block Sparse Bayesian Learning, ICASSP 2012
// %
// %   [2] Zhilin Zhang, Bhaskar D. Rao, Extension of SBL Algorithms for the 
// %       Recovery of Block Sparse Signals with Intra-Block Correlation, 
// %       available at: http://arxiv.org/abs/1201.0862
// %
// %   [3] webpage: http://dsp.ucsd.edu/~zhilin/BSBL.html, or
// %                https://sites.google.com/site/researchbyzhang/bsbl
// %
// % ============= Author =============
// %   Zhilin Zhang (z4zhang@ucsd.edu, zhangzlacademy@gmail.com)
// %
// % ============= Version =============
// %   1.4 (05/30/2012) make faster
// %   1.3 (05/28/2012)
// %   1.2 (01/22/2012)
// %   1.0 (08/27/2011)
// %

