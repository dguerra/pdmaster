/*
 * WavefrontSensor.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dailos
 */

#include "WavefrontSensor.h"
#include "CustomException.h"
#include "Zernikes.h"
#include "Zernikes.cpp"
#include <cmath>
#include "PDTools.h"
#include "TelescopeSettings.h"
#include "FITS.h"
#include "Metric.h"
#include "Minimization.h"

//ANEXO
//How to get with python null space matrix from constraints, Q2
//import scipy
//import scipy.linalg
//A = scipy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
//Q, R = scipy.linalg.qr(A)

constexpr double PI = 2*acos(0.0);

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
      //std::cout << "Input dataset images must be iqual size" << std::endl;
      throw CustomException("Input dataset images must be iqual size");
    }
  }

  TelescopeSettings tsettings(d_size.width);

  //c == recipients of zernike coefficients
  cv::Mat c = cv::Mat::zeros(14, 1, cv::DataType<double>::type);

  unsigned int pupilSideLength = optimumSideLength(d_size.width/2, tsettings.pupilRadiousPixels());
  std::cout << "pupilSideLength: " << pupilSideLength << std::endl;
  std::map<unsigned int, cv::Mat> zernikeCatalog = Zernikes<double>::buildCatalog(c.total(), pupilSideLength, tsettings.pupilRadiousPixels());

  std::cout << "Total original image energy: " << cv::sum(d.front()) << std::endl;

  std::vector<cv::Mat> D;
  for(cv::Mat di : d)
  {
    cv::Mat Di;
    cv::dft(di, Di, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    shift(Di, Di, Di.cols/2, Di.rows/2);
    D.push_back(Di);
  }

  double pupilRadiousP = tsettings.pupilRadiousPixels();
  cv::Mat pupilAmplitude = Zernikes<double>::phaseMapZernike(1, pupilSideLength, pupilRadiousP);
  
  Metric mm;
  Minimization minimizationKit;
//  std::vector<double> meanPowerNoise = {2.08519e-09, 1.9587e-09};    //sample case
  std::vector<cv::Mat> zBase = Zernikes<double>::zernikeBase(c.total(), pupilSideLength, pupilRadiousP);
  
  /*
  std::cout << "test begins" << std::endl;
  double data[] = {-2.401125838894388e-16, 0.3522232233399166, -0.379281849524244, 0.3534864099022184, 0.208522857754222, 0.7264457570573712, 0.3846718779804479, 0.09719520962238018, -0.5976538896757668, 0.2002786634782341, -0.03975816621627151, 0.09184171169366809, -0.08375642972811849, 0.06625904289356371, 2.401125838894387e-16, -0.3522232233399165, 0.3792818495242439, 0.3534864099022182, 0.208522857754222, 0.7264457570573709, 0.3846718779804478, 0.09719520962238014, -0.5976538896757666, 0.200278663478234, -0.0397581662162715, 0.09184171169366806, -0.08375642972811846, 0.0662590428935637};
                   

    cv::Mat pp(28, 1, cv::DataType<double>::type, data);
    
    Metric mp;
    mp.objectEstimate(pp, D, zBase, meanPowerNoise);
  
    cv::Mat oe = mp.F().clone();
    shift(oe, oe, oe.cols/2, oe.rows/2);  
    cv::idft(oe, oe, cv::DFT_REAL_OUTPUT);
  
    //writeFITS(oe, "../cosamia.fits");
  std::cout << "test ends"<< std::endl;
  */
  
  std::function<double(cv::Mat)> objFunction = std::bind(&Metric::objectiveFunction, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
    
  std::function<cv::Mat(cv::Mat)> gradFunction = std::bind(&Metric::gradient, &mm, std::placeholders::_1, D, zBase, meanPowerNoise);
      
  int M = zBase.size();
  int K = D.size();

  //constraints equation: PartlyKnownDifferencesInPhase
  cv::Mat ce = cv::Mat::eye(M, M, cv::DataType<double>::type);
  for(int i = 1; i < K-1; ++i)
  {
    cv::vconcat(ce, cv::Mat::eye(M, M, cv::DataType<double>::type), ce);
  }
  
  cv::hconcat(ce, -1 * cv::Mat::eye((K-1)*M, (K-1)*M, cv::DataType<double>::type), ce);
  ce.row(0) = cv::abs(ce.row(0));
  ce.row(1) = cv::abs(ce.row(1));
  ce.row(2) = cv::abs(ce.row(2));
  
  std::cout << "ce: " << std::endl << ce << std::endl;
  cv::Mat Q, R;
  
  householder(ce.t(), Q, R);
  std::cout << "Q: " << std::endl << Q << std::endl;
  // extracts A columns, 1 (inclusive) to 3 (exclusive).
  //Mat B = A(Range::all(), Range(1, 3));
  //Select last Np colums of Q to become the constraints null space
  int Np = (M*K) - ce.rows;
  cv::Mat Q2 = Q(cv::Range::all(), cv::Range(Q.cols - Np, Q.cols ));

  std::cout << "Q2: " << std::endl << Q2 << std::endl;
  //p: initial point; Q2: null space of constraints; objFunction: function to be minimized; gradFunction: first derivative of objFunction 
  cv::Mat p = cv::Mat::zeros(M*K, 1, cv::DataType<double>::type);
 
  minimizationKit.minimize(p, Q2, objFunction, gradFunction);
  std::cout << "mimumum: " << p.t() << std::endl;

  return c;
}

void WavefrontSensor::householder(const cv::Mat &m, cv::Mat &Q, cv::Mat &R)
{
  if(m.cols>m.rows) throw CustomException("Assertion failed: cols<rows for QR decomposition");
  auto matrix_minor = [](cv::Mat x, int d) -> cv::Mat
  { //For d = 2, turns this into this:
    // 5, 5, 5, 5, 5      1, 0, 0, 0, 0
    // 5, 5, 5, 5, 5      0, 1, 0, 0, 0
    // 5, 5, 5, 5, 5  ->  0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
  	cv::Mat m = cv::Mat::eye(x.size(), x.type());
  	for (int i = d; i < x.rows; i++)
    {
		  for (int j = d; j < x.cols; j++)
      {
	  		m.at<double>(i,j) = x.at<double>(i,j);
      }
    }
  
	  return m;
  };

  
	std::vector<cv::Mat> q;
	cv::Mat z = m.clone(), z1;
  
	for (int k = 0; k < m.cols && k < m.rows - 1; k++) 
  {
		cv::Mat e = cv::Mat::zeros(m.rows, 1, m.type()), x(m.rows, 1, m.type());
    double a;
		z1 = matrix_minor(z, k);
    
		z = z1.clone();
    z.col(k).copyTo(x);
    
		a = std::sqrt(x.dot(x));
    //If floating-point used, a should be opposite sign as the k-th coordinate m to avoid loss of significance
		if (m.at<double>(k,k) > 0) a = -a;  
		e.at<double>(k,0) = 1.0;
    e = x + (e*a);
		e = e / std::sqrt(e.dot(e));
    cv::Mat qi = cv::Mat::eye(e.rows, e.rows, e.type()) - 2.0 * e * e.t();
    
		q.push_back(qi.clone());
		z1 = qi * z;
		z1.copyTo(z);
	}
	
  //Q has dimensions of transpose of qi
  Q = cv::Mat::eye(q.front().cols, q.front().rows, q.front().type());
  //Multiply all elements of qi
  for(cv::Mat qi : q) Q = Q * qi.t(); 
  R = Q.t() * m;
}
  