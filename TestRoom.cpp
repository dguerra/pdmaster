/*
 * TestRoom.cpp
 *
 *  Created on: Feb 18, 2014
 *      Author: dailos
 */


#include <iostream>
#include <complex>
#include <cmath>
#include <random>
#include "TestRoom.h"
#include "ToolBox.h"
#include "Zernike.h"
#include "NoiseEstimator.h"
#include "Optics.h"
#include "FITS.h"
#include "SparseRecovery.h"
#include "OpticalSetup.h"
#include "ImageQualityMetric.h"
#include "ConvexOptimization.h"
#include <memory>


void test_jacobian()
{
  
  
}

//ML playgound:
//linear models
void leastSquearesExperiment()
{
  //We have N=15 samples from y=sin(x) in the interval [0,2π] and add gaussian noise
  unsigned int N = 5;  //Number of samples
  const double pi = 3.14159265358979323846;  /* pi */
  double startPoint(0.0);
  double endPoint(2.0 * pi);
  double stepSize((endPoint-startPoint)/double(N-1));
  cv::Mat x,t;
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::RNG& rng = cv::theRNG();
      
  for(double xi=startPoint; xi<=endPoint; xi=xi+stepSize)
  {
    x.push_back(xi);
    t.push_back(sin(xi)+rng.gaussian(0.2));
  }
  std::cout << "x: " << x.t() << std::endl;
  std::cout << "t: " << t.t() << std::endl;
  
  //Suppose now I want to modeled the system with a combination of M potentially nonlinear basis functions such as radial basis functions, RBF
  auto rbf = [](const double& x, const double& xm, const double& r) -> double
  { //exp(-(x-xm)^2/r^2)
    return std::exp(-std::pow(x-xm,2.0)/(r*r));
  }; 

  std::vector<cv::Mat> phi_v;
  for(double xm=startPoint; xm<=endPoint; xm=xm+stepSize)
  {
    cv::Mat phi_m;
    //We use the same locations for the center of the radial functions
    for(double xi=startPoint; xi<=endPoint; xi=xi+stepSize)
    {
      phi_m.push_back(rbf(xi,xm,1.0));
    }
    phi_v.push_back(phi_m);
  }
  cv::Mat phi;
  cv::hconcat(phi_v, phi);


  //Apply least squares to find the weights
  cv::Mat w = (phi.t()*phi+0.5*cv::Mat::eye(N,N,cv::DataType<double>::type)).inv()*phi.t()*t;   //Ridge regression
  //std::cout << "Ridge regression weights: " << w << std::endl;
  //cv::Mat w = (phi.t()*phi).inv()*phi.t()*t;   //Least squares

  std::vector<double> gamma_v(phi.cols, 1.0);
  w = perform_BSBL(phi, t, NoiseLevel::LittleNoise, gamma_v, 1);  //Noiseless, LittleNoise
  std::cout << "BSBL: " << w << std::endl;  
  
  gamma_v = std::vector<double>(phi.cols, 1.0);
  w = perform_SBL(phi, t, NoiseLevel::LittleNoise, gamma_v);  //Noiseless, LittleNoise
  std::cout << "SBL: " << w << std::endl;  
  
  std::cout << "END" << std::endl;  
  
/*  
  cv::Mat mu_x_old = cv::Mat::zeros(phi.cols, 1, cv::DataType<double>::type);
  cv::Mat Sigma_x, mu_x;
  cv::Mat gamma = cv::Mat::ones(phi.cols, phi.cols, cv::DataType<double>::type);
  double lambda(1e-3);
  
  for(unsigned int i=0;i<100;++i)
  {
    cv::Mat PhiGammPhiT = phi*gamma*phi.t();
    cv::Mat lambdaI(PhiGammPhiT.size(), cv::DataType<double>::type);
    cv::setIdentity(lambdaI, lambda);
      
    cv::Mat den = PhiGammPhiT + lambdaI;
    //Matrix right division should be implemented cv::solve instead of by inverse multiplication
    // xA = b  => x = b/A but never use x = b * inv(A)!!!
    //They are mathematically equivalent but not the same when working with floating numbers
    //x = cv.solve( A.t(), b.t() ).t();  equivalent to b/A
    //to perform matrix right division: mrdivide(b,A)
    cv::Mat H;
    cv::solve(den.t(), phi, H, cv::DECOMP_NORMAL);
    std::cout << "Hello" << std::endl;
      
    cv::Mat Hy = H.t() * t;
    cv::Mat HPhi = H.t() * phi;
      
    mu_x = gamma * Hy;
    
    cv::Mat diff = cv::abs(mu_x_old - mu_x);
    double maxVal;
    cv::minMaxLoc(diff, nullptr, &maxVal, nullptr, nullptr);
    if (maxVal < 1e-4)
    {
      std::cout << "Solution found." << std::endl;
      break;
    }
    
    Sigma_x = gamma - gamma * HPhi * gamma;  
      
    cv::Mat mu_x2;
    cv::multiply(mu_x, mu_x, mu_x2);
    gamma = Sigma_x + mu_x2;
    
    double l2 = cv::norm(t-(phi*mu_x), cv::NORM_L2);
    lambda = ((l2*l2) + cv::trace(phi*Sigma_x*phi.t()).val[0])/phi.rows;
  }

  w = mu_x.clone();
  std::cout << "Sparse Bayesian Learning weights: " << w << std::endl;
*/

/*
  //Show result: (prediction)
  cv::Mat result;
  cv::Mat cur;
  for(double cursor=startPoint;cursor<=endPoint;cursor=cursor+0.01)
  {
    double val(0.0);
    int i(0);
    for(double xm=startPoint; xm<=endPoint; xm=xm+stepSize)
    {
      val = val + w.at<double>(i++,0) * rbf(cursor,xm,1.0);
    }
    result.push_back(val);
    cur.push_back(cursor);
  }

  //std::cout << "phi: " << phi << std::endl;
  cv::Mat plot_data, plot_result;
  Plot2d plot1( x,t );
  Plot2d plot2( cur,result );
  //cv::plot->setPlotBackgroundColor( cv::Scalar( 50, 50, 50 ) ); // i think it is not implemented yet
  plot1.setPlotLineColor( cv::Scalar( 255, 255, 255 ) );
  plot1.setPlotLineWidth(2);
  plot1.setNeedPlotLine(false);
  plot1.render( plot_data );

  plot2.setPlotLineColor( cv::Scalar( 255, 255, 255 ) );
  plot2.setPlotLineWidth(2);
  plot2.setNeedPlotLine(false);
  plot2.render( plot_result );
  
  cv::Mat gray_result;
  cv::cvtColor(plot_result, gray_result, CV_BGR2GRAY);
  writeFITS(gray_result, "../gray_result.fits");
*/

/*
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window", plot_data );                   // Show our image inside it.
  cv::namedWindow( "Display window2", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window2", plot_result );                   // Show our image inside it.
  cv::waitKey();
*/
}


bool test_NoiseEstimator()
{
  //Benchmark
  cv::Mat img, dat;
  readFITS("../inputs/surfi000.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);
  cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
  std::cout << "cols: " << img.cols << " x " << "rows: " << img.rows << std::endl;

  //transfor to fourier domain and brings energy to the center
  cv::Mat D;
  cv::dft(img, D, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(D);

  //remove frequencies beyond cutoff
  OpticalSetup tsettings(D.cols);
  Zernike zrnk;
  D.setTo(0, zrnk.phaseMapZernike(1, D.cols, tsettings.cutoffPixel()) == 0);
  
  //Take back to image domain the remaing frequencies
  fftShift(D);
  cv::idft(D, img, cv::DFT_REAL_OUTPUT);
  
  //add gaussian noise
  cv::Mat noise(img.size(), cv::DataType<double>::type);
  cv::Scalar sigma(0.29), mean(0);
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::randn(noise, mean, sigma);
  cv::add(img, noise, img);

  //estimate the noise
  NoiseEstimator ne;
  ne.meanPowerSpectrum(img);

  std::cout << "Mean power: " << ne.meanPower() << std::endl;
  std::cout << "Sigma: " << ne.sigma() << std::endl;
  std::cout << "Sigma 2: " << ne.sigma2() << std::endl;
  return true;
}

//cv::randn(cv::Mat a, cv::Mat mean, cv::Mat std)
//fill matrix a with random values from normal distribution with mean vector and std matrix
template<class T>
cv::Mat createRandomMatrix(const unsigned int& xSize, const unsigned int& ySize)
{
  double sigma = 5.0;
  std::random_device rdevice;
  std::default_random_engine generator(rdevice());
  std::normal_distribution<> distribution(0, sigma);

  cv::Mat A = cv::Mat(xSize, ySize, cv::DataType<T>::type);
  for(auto it = A.begin<T>(); it != A.end<T>(); ++it)
  {
    (*it) = (T)distribution(generator);
  }
  return A;
}

bool test_minimization()
{
 
  ConvexOptimization mm;
  
  //write in wolfram alpha the following to verify: "minimize{x^8-3*(x+3)^5+5+(y+4)^6+y^5+3*z^2} in x+2*y+3*z=0"
  double ce_dat[] = {1.0, 2.0, 3.0};
  cv::Mat ce(1,3,cv::DataType<double>::type,ce_dat), Q, R;
  householder(ce.t(), Q, R);
  int numberOfUnkowns(3);
  int Np = numberOfUnkowns - ce.rows;
  cv::Mat Q2 = Q(cv::Range::all(), cv::Range(Q.cols - Np, Q.cols ));
  std::cout << "Q2: " << Q2 << std::endl;
  
  std::function<double(cv::Mat)> funcc = [] (cv::Mat x) -> double 
  {//Eq to minimize: x^8-3*(x+3)^5+5+(y+4)^6+y^5+3*z
    return std::pow(x.at<double>(0,0),8) - 3 * std::pow(x.at<double>(0,0)+3,5) + 5 + 
           std::pow(x.at<double>(1,0)+4,6)+ std::pow(x.at<double>(1,0),5) + 3 * std::pow(x.at<double>(2,0),2);
  };
  
  std::function<cv::Mat(cv::Mat)> dfuncc_diff = [funcc] (cv::Mat x) -> cv::Mat
  { //make up gradient vector through slopes and tiny differences
    double EPS(1.0e-4);
    cv::Mat df = cv::Mat::zeros(x.size(), x.type());
    for(unsigned int j = 0; j < x.total(); ++j)
    {
    	cv::Mat xh = x.clone();
      cv::Mat xl = x.clone();

      xh.at<double>(j,0) = xh.at<double>(j,0) + EPS;
      xl.at<double>(j,0) = xl.at<double>(j,0) - EPS;
      
      double fh = funcc(xh);
      double fl = funcc(xl);
      
      df.at<double>(j,0) = (fh-fl)/(2.0*EPS);
    }
    return df;
    
  };
  
  
  std::function<cv::Mat(cv::Mat)> dfuncc = [] (cv::Mat x) -> cv::Mat
  { //Gradient vector function: function derivative with every variable
    cv::Mat t(3,1, cv::DataType<double>::type);  //Size(3,1)->1 row, 2 colums
    t.at<double>(0,0) = 8 * std::pow(x.at<double>(0,0),7) - 15 * 
                            std::pow(x.at<double>(0,0)+3,4);
    t.at<double>(1,0) = 5 * std::pow(x.at<double>(1,0),4) + 6 * 
                            std::pow(x.at<double>(1,0)+4,5);
    t.at<double>(2,0) = 6 * x.at<double>(2,0);
    return t;
  };
 
  

  cv::Mat x0_conv = cv::Mat::zeros(3, 1, cv::DataType<double>::type);   //reset starting point
    
  //Lambda function that turn minimize function + constraints problem into minimize function lower dimension problem
  auto F_constrained = [] (cv::Mat x, std::function<double(cv::Mat)> func, const cv::Mat& Q2) -> double
  {
    return func(Q2*x);
  };
    
  auto DF_constrained = [] (cv::Mat x, std::function<cv::Mat(cv::Mat)> dfunc, const cv::Mat& Q2) -> cv::Mat
  {
    return Q2.t() * dfunc(Q2*x);
  };
      
  std::function<double(cv::Mat)> f_constrained = std::bind(F_constrained, std::placeholders::_1, funcc, Q2);
  std::function<cv::Mat(cv::Mat)> df_constrained = std::bind(DF_constrained, std::placeholders::_1, dfuncc_diff, Q2);
  //Define a new starting point with lower dimensions after reduction with contraints
  cv::Mat p_constrained = Q2.t() * x0_conv;

  mm.perform_BFGS(p_constrained, f_constrained, df_constrained);
  
  x0_conv = Q2 * p_constrained;   //Go back to original dimensional 

  std::cout << "mimumum: " << x0_conv.t() << std::endl;

  
  return true;
}

bool test_minQ2()
{
  //there are 4 parameters. The first is double than the third. The last should be zero
  double ce_dat[] = {1.0, 0.0, -1.0, 0.0};
  cv::Mat ce(1,4,cv::DataType<double>::type,ce_dat), Q, R;
  householder(ce.t(), Q, R);
  int numberOfUnkowns(4);
  int Np = numberOfUnkowns - ce.rows;
  cv::Mat Q2 = Q(cv::Range::all(), cv::Range(Q.cols - Np, Q.cols ));
  std::cout << "Q2: " << Q2 << std::endl;
  cv::Mat bias = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
  bias.at<double>(2, 0) = -2.0;
  cv::Mat p_constrained = cv::Mat::ones(3, 1, cv::DataType<double>::type);
  cv::Mat x0_conv = (Q2 * p_constrained) + bias;   //Go back to original dimensional 
  std::cout << "x cons: " << x0_conv << std::endl;
  
  p_constrained = (x0_conv.t() - bias.t()) * Q2;
  std::cout << "x uncons: " << p_constrained << std::endl;
  
  return true;
}


void test_SparseRecovery()
{
  double P_array[] = { -0.365062,   0.091490,   0.906873,  -0.880990,   0.346249,   0.249187,
                       -0.909919,   0.380777,  -0.413275,   0.060092,   0.927514,   0.639146,
                       -0.196918,   0.920130,  -0.082374,   0.469304,  -0.140821,  -0.727598 };
  cv::Mat P(3, 6, cv::DataType<double>::type, P_array);

  double s_array[] = {0.000000, 0.000000, -0.65563, 0.000000, 0.70438, 0.000000};
  cv::Mat s(6, 1, cv::DataType<double>::type, s_array);
  
  cv::Mat_<double> x = P * s;
  unsigned int sparsity = 2;
  //cv::Mat sol = perform_IHT(P, x, sparsity, 0.0);
  cv::Mat sol = perform_FISTA(P, x, 0.001);
  std::cout << "sol: " << sol.t() << std::endl;
}

bool test_BSL()
{
  unsigned int M = 80;          // row number of the dictionary matrix 
  unsigned int N = 164;          // column number

  unsigned int blkNum = 7;       // nonzero block number
  unsigned int blkLen = 2;       // block length

  double SNR = 80;         // Signal-to-noise ratio

  cv::Mat Phi(M, N, cv::DataType<double>::type);
  cv::randn(Phi, cv::Scalar(1.0), cv::Scalar(1.0));
  cv::Mat PhiPhi, sumPhiPhi, sqrtsumPhiPhi;
  cv::multiply(Phi, Phi, PhiPhi);
  cv::reduce(PhiPhi, sumPhiPhi, 0, CV_REDUCE_SUM);   //sum every column a reduce matrix to a single row
  cv::sqrt(sumPhiPhi, sqrtsumPhiPhi);
  cv::divide(Phi, cv::Mat::ones(M,1,cv::DataType<double>::type) * sqrtsumPhiPhi, Phi);

  unsigned int totalBlkNumber = N/blkLen;
  cv::Mat wgen = cv::Mat::zeros(totalBlkNumber, blkLen, cv::DataType<double>::type);
  
  cv::RNG rng(cv::getTickCount());
  for(unsigned int i=0; i<blkNum; ++i)
  {
    cv::Mat r(1, blkLen, cv::DataType<double>::type);
    randn(r, cv::Mat(1, 1, cv::DataType<double>::type, cv::Scalar(rng.uniform(50.0, 100.0)) ), 0.01 * cv::Mat::ones(1, 1, cv::DataType<double>::type));
    r.copyTo(wgen.row(i));
  }

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
  
  cv::Mat nwgen = shuffleRows(wgen);
  cv::Mat xgen = nwgen.reshape(0, nwgen.total());
  cv::Mat y = Phi * xgen;   //noiseless signal
  
  // Observation noise   
  cv::Scalar mean, stddev;
  cv::meanStdDev(y, mean, stddev);
  cv::Mat noise(y.size(), y.type());
  cv::randn(noise, cv::Scalar(0.0), cv::Scalar(stddev*std::pow(10,-SNR/20.0)));
  
  double Phi_array[] = { 0.889288,   0.727865,  -0.837231,  -0.246046,  -0.428118,  -0.812367,
                         0.148014,   0.256824,  -0.401575,   0.066427,  -0.464477,  -0.573846,
                         0.432734,   0.635809,  -0.371190,  -0.966979,   0.775227,   0.103731 };
                         
  cv::Mat nPhi(3, 6, cv::DataType<double>::type, Phi_array);
  double x_array[] = {0.00000, 0.00000,  0.00000,  0.00000, -1.68713, -1.40636};
  cv::Mat nx(6, 1, cv::DataType<double>::type, x_array);
  
  double y_array[] = { 1.8648, 1.5907, -1.4538};
  cv::Mat ny(3, 1, cv::DataType<double>::type, y_array);
  
  //OptAlg
  std::vector<double> gamma_v(3, 1.0);
  std::cout << "blkLen: " << blkLen << std::endl;
  std::cout << "nPhi: " << nPhi << std::endl;
  std::cout << "ny: " << ny << std::endl;
  cv::Mat res = perform_BSBL(nPhi, ny, NoiseLevel::Noiseless, gamma_v, blkLen);
  //cv::Mat res2 = perform_IHT(nPhi, ny, 2);
  
  
/*  
  for(unsigned int i=0;i<1;++i)
  {
    //gamma_v = std::vector<double>(3, 1.0);
    cv::Mat res = perform_BSBL(nPhi, nPhi*nx, NoiseLevel::Noiseless, gamma_v, 2);
    //cv::Mat res2 = perform_IHT(nPhi, nPhi*(10e-4*nx), 2);
    std::cout << "res: " << res.t() << std::endl;
    //std::cout << "res2: " << res2.t() << std::endl;
  }
*/  

  return true;
}

bool test_zernikes()
{
  cv::Mat Z5;
  double r_c(400.0);
  int side_l(1000);
  int z_max(10);
  std::shared_ptr<Zernike> zrnk = std::make_shared<Zernike>(r_c, side_l, z_max);

  unsigned int count_a(1);
  for(auto za : zrnk->base())
  {
    unsigned int count_b(1);
    for(auto zb : zrnk->base())
    {
      double za_l2 = cv::norm(za, cv::NORM_L2);
      double zb_l2 = cv::norm(zb, cv::NORM_L2);
      
      double inner_prod = za.dot(zb)/(za_l2*zb_l2);
      std::cout << "inner_prod: " << count_a << " con " << count_b << " -> " << inner_prod << std::endl;
      count_b++;
    }
    count_a++;
  }
  return true;
}


void test_covarianceMatrix()
{
  /*
  cv::Mat_<float> samples = (cv::Mat_<float>(4, 2) << 500.0, 500.0,
                                              355.8, 355.8,
                                              498.7, 498.7,
                                              123.4, 123.4 );
  
  */
  
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::RNG& rng = cv::theRNG();
  
  cv::Mat_<float> samples;
  
  for(unsigned int i=0;i<100000000;++i)
  {
    float val1 = rng.uniform(0.0, 1000.0);
    float val2 = rng.uniform(0.0, 1000.0);
    cv::Mat s = ( cv::Mat_<float>(1, 2) << val1, 234.4 );
    samples.push_back( s );
  }
  
  cv::Mat cov, mu;
  cv::calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
  
  cov = cov / (samples.rows - 1);
  
  std::cout << "cov: " << std::endl;
  std::cout << cov << std::endl;
  
  std::cout << "mu: " << std::endl;
  std::cout << mu << std::endl;

}

/*
void test_nonlinearCompressedSensing()
{
  //Test compressed sensing technique over a non-linear measurement process by using the jacobian
  std::function<double(cv::Mat)> fx = [] (cv::Mat x) -> double 
  { //fx = u^3-v^2
    return std::pow(x.at<double>(0,0),3) - std::pow(x.at<double>(1,0),2);
  };
  
  std::function<double(cv::Mat)> fy = [] (cv::Mat x) -> double 
  { //fy = u^3+v^2
    return std::pow(x.at<double>(0,0),3) + std::pow(x.at<double>(1,0),2);
  };
  
  std::function<cv::Mat(cv::Mat)> jf = [] (cv::Mat x) -> cv::Mat
  { //Jacobian matrix function: function derivative with every variable
    cv::Mat t(2,2, cv::DataType<double>::type);
    t.at<double>(0,0) = 3 * std::pow(x.at<double>(0,0),2);   //dfx/du
    
    t.at<double>(1,0) = 3 * std::pow(x.at<double>(0,0),2);   //dfy/du
                            
    t.at<double>(0,1) = -2.0 * x.at<double>(1,0);            //dfx/dv
    
    t.at<double>(1,1) =  2.0 * x.at<double>(1,0);            //dfy/dv
    return t;
  };
  
  cv::Mat x0 = cv::Mat::ones(2, 1, cv::DataType<double>::type);
  
  cv::Mat y = -(3*3);
}
*/

/*
bool test_nonsmoothConvexOptimization()
{
  ConvexOptimization mm;
  
  cv::Mat Q2 = cv::Mat::eye(2, 2, cv::DataType<double>::type);

  std::function<double(cv::Mat)> absXplusAbsY = [] (cv::Mat x) -> double 
  { //|x|+|y|
    return std::abs(x.at<double>(0,0)) + std::abs(x.at<double>(1,0));
  };
    //Subdifferential version
  std::function<cv::Mat(cv::Mat)> dAbsXplusAbsY = [] (cv::Mat x) -> cv::Mat
  { //Subdifferential vector function: subdifferential with every variable
    cv::Mat t(2,1, cv::DataType<std::complex<double> >::type);  //Size(2,1)->1 row, 2 colums
    auto sign = [](double a, double b) -> double {return b >= 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as positive sign
    auto sign_ = [](double a, double b) -> double {return b > 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as negative sign
    double x0 = x.at<double>(0,0);
    double x1 = x.at<double>(1,0);
    double dx0_h = sign(1.0, x0);
    double dx0_l = sign_(1.0, x0);
    double dx1_h = sign(1.0, x1);
    double dx1_l = sign_(1.0, x1);
    if(dx0_h < dx0_l) std::swap(dx0_h, dx0_l);
    t.at<std::complex<double> >(0,0) = std::complex<double>(dx0_l, dx0_h);
    if(dx1_h < dx1_l) std::swap(dx1_h, dx1_l);
    t.at<std::complex<double> >(1,0) = std::complex<double>(dx1_l, dx1_h);
    if(dx0_h != dx0_l || dx1_h != dx1_l) std::cout << "nonsmooth point." << std::endl;
    return t;
  };

  std::function<double(cv::Mat)> funcc = [] (cv::Mat xx) -> double 
  { //|x-1|+|y-3|+x^2
    double x = xx.at<double>(0,0);
    double y = xx.at<double>(1,0);
    return std::abs(x-1) + std::abs(y-3) + x*x;
  };
  std::function<cv::Mat(cv::Mat)> dfuncc = [] (cv::Mat xx) -> cv::Mat
  { //Subdifferential vector function: subdifferential with every variable
    auto sign = [](double a, double b) -> double {return b >= 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as positive sign
    auto sign_ = [](double a, double b) -> double {return b > 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as negative sign
    cv::Mat t(2,1, cv::DataType<std::complex<double> >::type);  //Size(2,1)->1 row, 2 colums
    double x = xx.at<double>(0,0);
    double y = xx.at<double>(1,0);
    double dx_h = sign(1.0,x-1.0) + 2 * x;
    double dx_l = sign_(1.0,x-1.0) + 2 * x;

    double dy_h = sign(1.0, y-3.0);
    double dy_l = sign_(1.0, y-3.0);
    if(dx_h != dx_l || dy_h != dy_l) std::cout << "nonsmooth point." << std::endl;
    if(dx_l>dx_h) std::swap(dx_l, dx_h);
    t.at<std::complex<double> >(0,0) = std::complex<double>(dx_l, dx_h);
    if(dy_l>dy_h) std::swap(dy_l, dy_h);
    t.at<std::complex<double> >(1,0) = std::complex<double>(dy_l, dy_h);

    return t;
  };

  std::function<double(cv::Mat)> onedim = [] (cv::Mat xx) -> double 
  { //|x|
    double x = xx.at<double>(0,0);
    return std::abs(x-4);
  };
  std::function<cv::Mat(cv::Mat)> donedim = [] (cv::Mat xx) -> cv::Mat
  { //Subdifferential vector function: subdifferential with every variable
    auto sign = [](double a, double b) -> double {return b >= 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as positive sign
    auto sign_ = [](double a, double b) -> double {return b > 0.0 ? std::abs(a) : -std::abs(a);};   //Consider zero as negative sign
    cv::Mat t(1,1, cv::DataType<std::complex<double> >::type);  //Size(2,1)->1 row, 2 colums
    double x = xx.at<double>(0,0);
    double dx_h = sign(1.0,x-4);
    double dx_l = sign_(1.0,x-4);

    if(dx_h != dx_l) std::cout << "nonsmooth point." << std::endl;
    if(dx_l>dx_h) std::swap(dx_l, dx_h);
    t.at<std::complex<double> >(0,0) = std::complex<double>(dx_l, dx_h);

    return t;
  };
  
  cv::Mat p = cv::Mat::zeros(2, 1, cv::DataType<double>::type);// + 3.9 * cv::Mat::ones(2, 1, cv::DataType<double>::type);
  p.at<double>(0,0) = -13;
  p.at<double>(1,0) = -5;
  //mm.minimize(p, Q2, funcc, dfuncc);
  mm.minimize(p, Q2, absXplusAbsY, dAbsXplusAbsY);
  
  std::cout << "p " << p.t() << std::endl;
  std::cout << "fret " << mm.fret() << std::endl;
  
  return true; 
}

*/

bool test_convolveDFT_vs_crosscorrelation()
{
  bool full(true), corr(true);
  cv::Mat B1, B2;
  cv::Mat A_real = createRandomMatrix<double>(5,5);
  cv::Mat A_imag = createRandomMatrix<double>(5,5);
  cv::Mat planes[2] = {A_real, A_imag};
  cv::Mat A;
  cv::merge(planes, 2, A);
  convolveDFT(A, A, B1, full, corr);
  //cv::copyMakeBorder(A, A, Top, Bottom, Left, Right, cv::BORDER_CONSTANT);
  cv::copyMakeBorder(B1, B1, 1, 0, 1, 0, cv::BORDER_CONSTANT);
  fftShift(B1);
  
  B2 = crosscorrelation(A, A);
  std::cout << "B1" << splitComplex(B1).first << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "B2" << splitComplex(B2).first << std::endl;
  
  return true;
}

/*
void test_udwd_spectrums()
{
  cv::Mat dat, input;
  readFITS("../inputs/pd.004.fits", dat);
  dat.convertTo(input, cv::DataType<double>::type);
  
  cv::Mat in = input(cv::Rect(cv::Point(0,0), cv::Size(450,450))).clone();
  if(! in.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    //return -1;
  }
  std::vector<cv::Mat> output;
  cv::Mat residu;
  cv::Mat in_spec;
  
  cv::dft(in, in_spec, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
//  fftShift(in_spec);
  
  swtSpectrums(in_spec, output, residu, 7);
  
  cv::Mat wSum = cv::Mat::zeros(output.back().size(), output.back().type());
  cv::Mat o_tmp;
  for(cv::Mat it : output)
  {
    wSum += it;
  }

  wSum = wSum + residu;
  cv::Mat wSum_measure;
//  fftShift(wSum);
  cv::idft(wSum, wSum_measure, cv::DFT_REAL_OUTPUT); 
    
  cv::Mat diff = in - wSum_measure;
  writeFITS(wSum_measure, "../d1.fits");
  std::cout << "Energy difference: " << std::sqrt(cv::sum(diff.mul(diff)).val[0]) << std::endl;  
}

void test_generizedPupilFunctionVsOTF()
{
  double diversityFactor_ = -2.21209;
  constexpr double PI = 2*acos(0.0);
  double c4 = diversityFactor_ * PI/(2.0*std::sqrt(3.0));
  Zernike zrnk;
  cv::Mat z4 = zrnk.phaseMapZernike(4, 128, 50);
  double z4AtOrigen = z4.at<double>(z4.cols/2, z4.rows/2);
  cv::Mat pupilAmplitude = zrnk.phaseMapZernike(1, 128, 50);
  cv::Mat c = cv::Mat::zeros(14, 1, cv::DataType<double>::type);
  c.at<double>(0,3) = 0.8;
  c.at<double>(0,4) = 0.3;
  c.at<double>(0,6) = 0.5;
  c.at<double>(0,9) = 0.02;

  cv::Mat c1 = cv::Mat::zeros(14, 1, cv::DataType<double>::type);
  c1.at<double>(0,3) = 0.7;
  c1.at<double>(0,5) = 0.23;
  c1.at<double>(0,6) = 0.9;
  c1.at<double>(0,10) = 0.42;

  cv::Mat focusedPupilPhase = zrnk.phaseMapZernikeSum(128, 50, c);
  cv::Mat focusedPupilPhase1 = zrnk.phaseMapZernikeSum(128, 50, c1);

  cv::Mat defocusedPupilPhase = focusedPupilPhase + c4*(z4-z4AtOrigen);
  cv::Mat defocusedPupilPhase1 = focusedPupilPhase1 + c4*(z4-z4AtOrigen);


  Optics focusedOS(focusedPupilPhase, pupilAmplitude);
  Optics focusedOS1(focusedPupilPhase1, pupilAmplitude);
  Optics defocusedOS(defocusedPupilPhase, pupilAmplitude);
  Optics defocusedOS1(defocusedPupilPhase1, pupilAmplitude);
  //cv::Mat result = divComplex(focusedOS.otf(),defocusedOS.otf());
  cv::Mat result = focusedOS.generalizedPupilFunction()-defocusedOS.generalizedPupilFunction();
  cv::Mat result1 = focusedOS1.generalizedPupilFunction()-defocusedOS1.generalizedPupilFunction();
  showComplex(result, "res", false, false);
  showComplex(result1, "res1", false, false);

}

void test_wavelet_zernikes_decomposition()
{
  cv::Mat input = cv::Mat_<double>(cv::imread("/home/dailos/workspace/fruits.jpg", CV_LOAD_IMAGE_GRAYSCALE));
  cv::Mat in = input(cv::Rect(cv::Point(0,0), cv::Size(450,450))).clone();
  if(! in.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    //return -1;
  }
  std::vector<cv::Mat> output;
  cv::Mat residu;

  udwd(in, output, residu, 7);
}


void test_zernike_wavelets_decomposition()
{
  std::vector<cv::Mat> vCat;
  Zernike zrnk;
  std::map<unsigned int, cv::Mat> cat = zrnk.buildCatalog(20, 200, 200/2);
  std::vector<cv::Mat> wavelet_planes;
  cv::Mat residu;
  unsigned int count(0);
  for(std::pair<unsigned int, cv::Mat> i : cat )
  {
    if(count > 0)
    {
    wavelet_planes.clear();
    udwd(i.second, wavelet_planes, residu, 9);
    vCat.push_back(i.second);
    vCat.insert( vCat.end(), wavelet_planes.begin(), wavelet_planes.end() );
    vCat.push_back(residu);
    }
    ++count;
  }


  cv::Mat can = makeCanvas(vCat, 70*13, 20);

  cv::imshow("zer", can);
  cv::waitKey();
}

void test_noiseFilter()
{
  unsigned int ncols(5), nrows(5);
  cv::Mat H = createRandomMatrix<double>(ncols,nrows);
  double filter_upper_limit(4.0);
  double filter_lower_limit(0.1);
  std::cout << "H: " << std::endl << H << std::endl;

  H.setTo(0, H < filter_lower_limit);
  std::cout << "H set to 0 below lower: " << std::endl << H << std::endl;
  H.setTo(filter_upper_limit, H > filter_upper_limit);
  std::cout << "H set to upper above upper: " << std::endl << H << std::endl;
}

bool test_conjComplex()
{
  unsigned int ncols(5), nrows(5);
  cv::Mat A = makeComplex(createRandomMatrix<double>(ncols,nrows),createRandomMatrix<double>(ncols,nrows));
  cv::Mat B;
  std::cout << "Matrix A: " << std::endl;
  std::cout << A << std::endl;
  cv::mulSpectrums(A,conjComplex(A),B, cv::DFT_COMPLEX_OUTPUT);
  std::cout << "A multiplied by A*: " << std::endl;
  cv::Mat firstResult = (splitComplex(B)).first;
  std::cout << firstResult << std::endl;
  std::cout << "Squared modulus of A: " << std::endl;
  cv::Mat secondResult = (absComplex(A)).mul(absComplex(A));
  std::cout << secondResult << std::endl;
  std::cout << "Per-element comparison between the two results: " << std::endl;
  cv::Mat cmp = (firstResult == secondResult);
  std::cout << cmp << std::endl;
  return true;
}

void test_selectCentralROI()
{
  cv::Mat o = cv::Mat::zeros(10, 20, cv::DataType<double>::type);
  float m[] = { 1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16 };
  cv::Mat kernelLoG(5, 1, CV_32F, m);
  //cv::Mat kernel = kernelLoG * kernelLoG.t();
  kernelLoG.copyTo(selectCentralROI(o, kernelLoG.size()));
  //cv::dft(o, o, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  std::cout << o << std::endl;
}

void test_AWMLE()
{
  cv::Mat psf, img, object;
  readFITS("/home/dailos/workspace/psf_SAA_SR_51.fits", psf);
  readFITS("/home/dailos/workspace/saturno_x_SR53_B_+_Gau01.0.fits", img);

  cv::Mat psf_norm, img_norm;
  cv::normalize(psf, psf_norm, 0, 1, CV_MINMAX);
  cv::normalize(img, img_norm, 0, 1, CV_MINMAX);
  cv::imshow("psf", psf_norm);
  cv::imshow("img", img_norm);
  cv::waitKey();

  if(!img.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    //return -1;
  }

  double sigmaNoise = 1.0;
  AWMLE(img_norm, psf_norm, object, sigmaNoise, 7);
}

void test_wavelets()
{
  cv::Mat input = cv::Mat_<double>(cv::imread("/home/dailos/workspace/fruits.jpg", CV_LOAD_IMAGE_GRAYSCALE));
  cv::Mat in = input(cv::Rect(cv::Point(0,0), cv::Size(450,450))).clone();
  if(! in.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    //return -1;
  }
  std::vector<cv::Mat> output;
  cv::Mat residu;

  udwd(in, output, residu, 7);
  //swt(in, output, residu, 7);
  cv::Mat wSum = cv::Mat::zeros(output.back().size(), output.back().type());
  cv::Mat o_tmp;
  for(cv::Mat it : output)
  {
    wSum += it;
  }

  wSum = wSum + residu;
  cv::Mat diff = in - wSum;
  cv::normalize(wSum, wSum, 0, 1, CV_MINMAX);
  cv::imshow("wavelet Sum", wSum);
  cv::Mat can = makeCanvas(output, 300, 1);
  cv::imshow("can", can);
  cv::waitKey();
  std::cout << "Energy difference: " << std::sqrt(cv::sum(diff.mul(diff)).val[0]) << std::endl;
}

void test_conv_flaw()
{
  cv::Mat out;
  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0};
  int n = sizeof(data) / sizeof(data[0]);
  cv::Mat row(n, 1, cv::DataType<double>::type, data);
  cv::Mat kernel = cv::getGaussianKernel(3, -1);
  std::cout << "kernel: " << kernel*kernel.t() << std::cout;
  std::cout << "row*row: " << row*row.t() << std::endl;
  conv_flaw(row*row.t(), kernel*kernel.t(), out);
  std::cout << "out: " << out << std::endl;
}

void test_divComplex()
{
  std::complex<float> c(73,12);
  std::complex<float> d(67,9);
  cv::Mat cMat(1,1,cv::DataType<std::complex<float> >::type);
  cv::Mat dMat(1,1,cv::DataType<std::complex<float> >::type);
  cMat.at<std::complex<float> >(0,0) = c;
  dMat.at<std::complex<float> >(0,0) = d;
  std::complex<float> r = c/d;
  cv::Mat rMat = divComplex(cMat,dMat);
  std::cout << "r: " << r << std::endl;
  std::cout << "rMat: " << rMat << std::endl;

}

void test_phaseMapZernikeSum()
{
  double data[] = {0, 0, 0, -0.2207034660012752, -0.2771620624276502, -0.451531165841092,
                   -0.3821562081878558, 0.275334782691961, 0.2975674509517756, 0.01384845253351654};
  //double data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5}; //0.5, 0.2, 0.4, 0.8, 0.7, 0.6};
  int n = sizeof(data) / sizeof(data[0]);
  cv::Mat coeffs(n, 1, CV_64FC1, data);
  std::cout << "coeffs: " << coeffs << std::endl;
  Zernike zrnk;
  cv::Mat phaseMapSum = zrnk.phaseMapZernikeSum(136/2, 32.5019, coeffs);
  std::cout << "phaseMapSum.at<double>(30,30): " << phaseMapSum.at<double>(40,40) << std::endl;

  for(unsigned int i=4;i<=10;++i)
  {
    cv::Mat zern = zrnk.phaseMapZernike(i, 136/2, 32.5019);
    cv::normalize(zern, zern, 0, 1, CV_MINMAX);
    writeOnImage(zern, std::to_string(i));
    cv::imshow(std::to_string(i), zern);
  }

  //cv::normalize(phaseMapSum, phaseMapSum, 0, 1, CV_MINMAX);
  //cv::imshow("phaseMapSum", phaseMapSum);
  cv::waitKey();
}

bool test_specular()
{
  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
  cv::Mat a(8, 8, CV_8UC1, data);
  cv::Mat b,c;
  cv::flip(a,b,-1);
  shift(b,c,1,1);
  std::cout << "Specular: " << c << std::endl;
  return true;
}

bool test_normalization()
{
  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3,
                              1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
  cv::Mat a(8, 8, CV_8UC1, data);

  cv::Mat b = makeComplex(cv::Mat_<float>(a), createRandomMatrix<float>(8,8));
  cv::Mat normB;
  normComplex(b,normB);
  std::cout << normB << std::endl;
  return true;
}

void test_Optics()
{
  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3,
                                1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
  cv::Mat p(8, 8, CV_8UC1, data);
  cv::Mat a = cv::Mat::ones(p.size(), p.type());
  Optics os = Optics(cv::Mat_<float>(p),cv::Mat_<float>(a));

  std::cout << "otf():" << std::endl;
  std::cout << os.otf() << std::endl;
  std::cout << "generalizedPupilFunction():" << std::endl;
  std::cout << os.generalizedPupilFunction() << std::endl;
}

void test_Noise()
{
  cv::Mat img = cv::Mat::zeros(100,100,CV_32F);
  cv::theRNG() = cv::RNG( time (0) );
  cv::Mat noise(img.size(), CV_32F);
  cv::Scalar s_(20), m_(0);
  cv::randn(noise, m_, s_);
  cv::Mat blank(img.size(), CV_32F);
  cv::Mat tmpImg(img.size(), CV_32F);
  img.convertTo(tmpImg, CV_32F);

  blank = cv::Mat::zeros(img.size(), CV_32F) + cv::Scalar(50) + noise;
  //blank = tmpImg + noise;
  //double mean, sigma;
  NoiseEstimator estimateNoise;
  estimateNoise.kSigmaClipping(blank);
  //cout << "mean: " << mean << endl;
  //cout << "sigma: " << sigma << endl;
  cv::imshow("original", tmpImg);
  cv::imshow("noise", noise);
  cv::imshow("Noisy image", blank);
}

void test_getNM()
{
  unsigned int N;
  int M;
  Zernike zrnk;
  for(unsigned int j=1; j<=10; ++j)
  {
    zrnk.getNM(j,N,M);
    std::cout << "Zernike Index: " << j << "; N: " << N << "; M: " << M << std::endl;
  }
}

void test_flip()
{
  unsigned int ncols(5), nrows(5);
  cv::Mat A = makeComplex(createRandomMatrix<float>(ncols,nrows),createRandomMatrix<float>(ncols,nrows));
  std::cout << "A(1,1): " << A.at<std::complex<float> >(1,1) << std::endl;
  std::cout << "A: " << std::endl << A << std::endl;

  cv::flip(A, A, -1);
  std::cout << "A flipped: " << std::endl << A << std::endl;

  shift(A, A, 1, 1);
  std::cout << "A(1,1): " << A.at<std::complex<float> >(1,1) << std::endl;
  std::cout << "A flipped shifted: " << std::endl << A << std::endl;

  cv::Mat s = conjComplex(A);
  std::cout << "A(1,1): " << s.at<std::complex<float> >(1,1) << std::endl;
  std::cout << "A flipped shifted: " << std::endl << s << std::endl;

}

void test_ErrorMetric()
{
  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3,
                                  1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
  cv::Mat pa(8, 8, CV_8UC1, data);
  cv::Mat amp = cv::Mat::ones(pa.size(), pa.type());
  cv::Mat pb;
  cv::flip(pa,pb,-1);
  Optics focusedOS = Optics(cv::Mat_<float>(pa),cv::Mat_<float>(amp));
  Optics defocusedOS = Optics(cv::Mat_<float>(pb),cv::Mat_<float>(amp));


  cv::Mat absT0 = absComplex(focusedOS.otf());
  cv::Mat absTk = absComplex(defocusedOS.otf());
  double qFineTuning(0.0);   //additive constant for Q, adding offset, (not needed by now)
  cv::Mat Q = absT0.mul(absT0) + 1.0 * absTk.mul(absTk) + qFineTuning;
    //1/(sqrt(tmp)>1.0e-35)*tsupport  ????
  cv::pow(Q, -1./2, Q);   //Q is a real matrix
  std::cout << "absT0: " << std::endl;
  std::cout << absT0 << std::endl;
  std::cout << "sum(absT0): " << std::endl;
  std::cout << cv::sum(absT0).val[0] << std::endl;
  std::cout << "absTk: " << std::endl;
  std::cout << absTk << std::endl;

  std::cout << "Q: " << std::endl;
  std::cout << Q << std::endl;
}

void test_SVD()
{
//  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3,
//                              1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
//  cv::Mat a(8, 8, CV_8UC1, data);

  cv::Mat a = createRandomMatrix<double>(8,8);
  cv::Mat w, u, vt, ws;
  cv::SVD::compute(a, w, u, vt, cv::SVD::FULL_UV);
  double maxVal(0.0);
  cv::minMaxIdx(w, nullptr, &maxVal);

  std::cout << maxVal << std::endl;
  double singularityThreshold = maxVal*0.1;

  w.copyTo(ws,w>singularityThreshold);

  std::cout << "ws:" << std::endl;
  std::cout << ws << std::endl;

  cv::Mat sv = cv::Mat::diag(ws);

  cv::Mat result = u * sv * vt;

  std::cout << "a:" << std::endl;
  std::cout << a << std::endl;
  std::cout << "u:" << std::endl;
  std::cout << u << std::endl;
  std::cout << "sv:" << std::endl;
  std::cout << sv << std::endl;
  std::cout << "vt:" << std::endl;
  std::cout << vt << std::endl;

  std::cout << "result should be equal to 'a':" << std::endl;
  std::cout << result << std::endl;

  cv::Mat C;
  cv::SVD::backSubst(ws, u, vt, createRandomMatrix<double>(8,1), C);
  cv::Scalar m, s;
  cv::meanStdDev(C, m, s);
  std::cout << s.val[0] << std::endl;
  std::cout << std::sqrt(cv::sum((C-m.val[0]).mul(C-m.val[0])).val[0]) << std::endl;


}


bool test_fourier()
{  //check with idl results
  unsigned char data[8][8] = {1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3,
                              1,2,3,4,5,4,3,2,   2,3,4,5,6,5,4,3,   3,4,5,6,7,6,5,4,   2,3,4,5,6,5,4,3};
  cv::Mat a(8, 8, CV_8UC1, data);
  cv::Mat b,c;
  //cv::dft(cv::Mat_<float>(a),b,cv::DFT_COMPLEX_OUTPUT);
  b = crosscorrelation_direct(makeComplex(cv::Mat_<float>(a)),makeComplex(cv::Mat_<float>(a)));
  c = crosscorrelation(makeComplex(cv::Mat_<float>(a)),makeComplex(cv::Mat_<float>(a)));
  cv::Mat normC;
  normComplex(c,normC);
  std::cout << "Cross: " << normC << std::endl;
  //std::cout << "Cross: " << b/c << std::endl;
  return true;
}

bool test_shift()
{
  unsigned int ncols(9), nrows(5);
  cv::Mat A = makeComplex(createRandomMatrix<float>(ncols,nrows),createRandomMatrix<float>(ncols,nrows));
  cv::Mat aPad;
  cv::copyMakeBorder(A, aPad, 0, 3, 0, 6, cv::BORDER_CONSTANT, cv::Scalar(0.0));
  std::cout << "Matrix A: " << std::endl;
  std::cout << aPad << std::endl;
  cv::Mat B;
  //shift(A,A,1,3);
  shift(aPad,aPad,1,3);
  std::cout << "Matrix A shifted: " << std::endl;
  std::cout << aPad << std::endl;
  return true;
}

void test_filter2D()
{
  double m[] = {1.0, 2.0, 4.0, 6.0};
  cv::Mat row_ref(4, 1, cv::DataType<double>::type, m);
  cv::Mat kernel = row_ref * row_ref.t();
  cv::Mat source = createRandomMatrix<double>(10,10);
  cv::Mat out1, out2;
  cv::Point anchor = cv::Point(2, 2);
  cv::Mat source_padded;
  cv::copyMakeBorder(source, source_padded, kernel.rows, kernel.rows, kernel.cols, kernel.cols, cv::BORDER_CONSTANT);

  cv::filter2D(source, out1, source.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
  cv::filter2D(source_padded, out2, source_padded.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
  std::cout << source << std::endl;
  std::cout << out1 << std::endl;
  std::cout << source_padded << std::endl;
  std::cout << out2 << std::endl;
}

void test_convolution_algo()
{
  //double m[] = {1.0, 4.0, 1.0, 5.0, 1.0};
  double m[] = {0.0, 0.0, 2.0, 0.0, 0.0};
  cv::Mat row_ref(5, 1, cv::DataType<double>::type, m);
  cv::Mat kernel = row_ref * row_ref.t();

  cv::Mat imgOriginal;
  imgOriginal = createRandomMatrix<double>(10,10);
//  cv::Mat o;
//  convolve(imgOriginal, kernel, o, true, true);
//  std::cout << "original: " << std::endl;
//  std::cout << imgOriginal << std::endl;
//  std::cout << "result: " << std::endl;
//  std::cout << o << std::endl;

  cv::Mat outtt1, outtt2, outft1, outft2, outtf1, outtf2, outff1, outff2;

  convolveDFT(imgOriginal, kernel, outft1, false, true);
  convolve(imgOriginal, kernel, outft2, false, true);
  bool c1 = cv::checkRange(outft1-outft2, true, nullptr, -0.00001, 0.00001 );
  std::cout << c1 << std::endl;

  convolveDFT(imgOriginal, kernel, outtt1, true, true);
  convolve(imgOriginal, kernel, outtt2, true, true);
  bool c2 = cv::checkRange(outtt1-outtt2, true, nullptr, -0.00001, 0.00001 );
  std::cout << c2 << std::endl;

  convolveDFT(imgOriginal, kernel, outtf1, true, false);
  convolve(imgOriginal, kernel, outtf2, true, false);
  bool c3 = cv::checkRange(outtf1-outtf2, true, nullptr, -0.00001, 0.00001 );
  std::cout << c3 << std::endl;

  convolveDFT(imgOriginal, kernel, outff1, false, false);
  convolve(imgOriginal, kernel, outff2, false, false);
  bool c4 = cv::checkRange(outtf1-outtf2, true, nullptr, -0.00001, 0.00001 );
  std::cout << c4 << std::endl;
 }

void test_QualityMetric()
{
  cv::Mat a = createRandomMatrix<double>(10,10);
  //a = a + 29;

  cv::Mat b = createRandomMatrix<double>(10,10);
  //b = b * 40;
  ImageQualityMetric qm;

  //correlation coefficient
  double res1 = qm.correlationCoefficient(a, b);
  cv::Mat res2;
  cv::matchTemplate(cv::Mat_<float>(a), cv::Mat_<float>(b), res2, CV_TM_CCOEFF_NORMED);
  std::cout << (res1) << std::endl;
  std::cout << res2.at<float>(0,0) << std::endl;

  //meanSquareError
  double mse1 = qm.meanSquareError(a, b);
  cv::Mat mse2;
  cv::matchTemplate(cv::Mat_<float>(a), cv::Mat_<float>(b), mse2, CV_TM_SQDIFF);
  std::cout << (mse1) << std::endl;
  std::cout << mse2/a.total() << std::endl;

  double covar1 = qm.covariance(a, b);
  cv::Mat covar2;
  cv::matchTemplate(cv::Mat_<float>(a), cv::Mat_<float>(b), covar2, CV_TM_CCOEFF);
  std::cout << covar1 << std::endl;
  std::cout << covar2/a.total() << std::endl;

}

void test_convolve()
{
  unsigned char data[4][4] = {1,2,3,4,   2,3,4,5,   3,4,5,6,   4,5,6,7};
  cv::Mat A(4, 4, CV_8UC1, data);
  cv::Mat B = A*2;
  cv::Mat fftA, fftB;
  std::cout << "Hello" << std::endl;
  cv::dft(cv::Mat_<double>(A),fftA,cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  std::cout << "fftA: " << fftA << std::endl;

  cv::dft(cv::Mat_<double>(B),fftB,cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  std::cout << "fftB: " << fftB << std::endl;
  cv::Mat conv;
  convolve(fftA, fftB, conv);
  std::cout << conv << std::endl;
}

bool test_crosscorrelation()
{
  //unsigned int ncols(8), nrows(8);
  //cv::Mat A = createRandomMatrix<float>(ncols,nrows);
  //cv::Mat B = createRandomMatrix<float>(ncols,nrows);

  unsigned char data[4][4] = {1,2,3,4,   2,3,4,5,   3,4,5,6,   4,5,6,7};
  cv::Mat A(4, 4, CV_8UC1, data);
  cv::Mat B = A*2;
  cv::Mat fftA, fftB;
  cv::dft(cv::Mat_<float>(A),fftA,cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  std::cout << "fftA: " << fftA << std::endl;

  cv::dft(cv::Mat_<float>(B),fftB,cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  std::cout << "fftB: " << fftB << std::endl;

  cv::Mat cross = crosscorrelation(fftA, fftB);
  //cv::Mat conv = crosscorrelation_direct(A,B);

  std::cout << "crosscorrelation: " << std::endl;
  std::cout << cross << std::endl;
  //std::cout << "crosscorrelation_direct: " << std::endl;
  //std::cout << conv << std::endl;
  return true;
}

*/