/*
 * CompressedSensing.cpp
 *
 *  Created on: Sep 01, 2015
 *      Author: dailos
 */

#include "CompressedSensing.h"
#include "CustomException.h"
#include "FITS.h"

void dctMatrix(const unsigned int& m, const unsigned int& n, cv::Mat& mat)
{
  if(m>n) throw CustomException("Number of columns must be larger than rows.");
  //create matrix A mxn with a cosine frequency at every column
  cv::Mat src = cv::Mat::eye(n, m, cv::DataType<double>::type);
  cv::idct(src, mat, cv::DCT_ROWS);
  //cv::Rect rect(0, 0, m, n);
  //mat = mat(rect).clone();
}

void matchingPursuit()
{
  unsigned int m(10), n(30);
  //create matrix A mxn with a cosine frequency at every column
  cv::Mat At;
  dctMatrix(m,n,At);
  
  //Desing a sparse vector x with olny few nonzero coefficients
  cv::Mat x(n, 1, cv::DataType<double>::type);
  x.at<double>(0, 2) = 2.3;
  x.at<double>(0, 6) = 1.0;
  x.at<double>(0, 13) = 5.0;
  
  //L2 normalize every column of A????
  
  //Create our meassurement vector y
  cv::Mat y = At.t() * x;
  cv::Mat r = y.clone();   //Initilize residual
  cv::Mat x0 = cv::Mat::zeros( x.size(), x.type() );
  
  //Start MP algorithm
  for(unsigned int i=0; i<10; ++i)
  {  
    cv::Mat g = At * r;
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(g, nullptr, &maxVal, nullptr, &maxLoc);
    //Update x0 estimate as well as r residual
    r = r - maxVal * At.t().col( maxLoc.y );
    x0.at<double>( maxLoc.y ) += maxVal;
  }
  std::cout << "x0: " << x0 << std::endl;
}

void iterativeHardThresholding()
{
 
  auto randomMatrix = [](const unsigned int& xSize, const unsigned int& ySize) -> cv::Mat
  {
    double sigma = 5.0;
    std::random_device rdevice;
    std::default_random_engine generator(rdevice());
    std::normal_distribution<> distribution(0, sigma);

    cv::Mat A = cv::Mat(xSize, ySize, cv::DataType<double>::type);
    for(auto it = A.begin<double>(); it != A.end<double>(); ++it)
    {
      (*it) = (double)distribution(generator);
    }
    return A;
  };
  
  auto matrixToVector = [](const cv::Mat &matrix) -> cv::Mat
  {
    cv::Mat vector = matrix.col(0);
    for(unsigned int i = 1; i<matrix.cols; ++i)
    {
      cv::vconcat(vector, matrix.col(i), vector);
    }
    return vector;
  };
  
  auto vectorToMatrix = [](const cv::Mat &vector, const unsigned int& M) -> cv::Mat
  {
    if(vector.cols != 1 || vector.total() % M != 0) throw CustomException("Wrong vector format.");
    unsigned int N = vector.total() / M;
    cv::Mat matrix = vector(cv::Range(0, M), cv::Range::all());
    for(unsigned int i=1;i<N; ++i)
    {
      cv::hconcat(matrix, vector(cv::Range(i*M, i*M + M), cv::Range::all()), matrix);
    }
    return matrix;
  };
  
  auto shuffleRows = [](const cv::Mat &matrix) -> cv::Mat
  {
    
    std::vector <int> seeds;
    for (int cont = 0; cont < matrix.rows; cont++)
    {
      seeds.push_back(cont);
    }
    cv::theRNG() = cv::RNG( time (0) );
    cv::randShuffle(seeds);

    cv::Mat output;
    for (int cont = 0; cont < matrix.rows; cont++)
    {
      output.push_back(matrix.row(seeds[cont]));
    }
    return output;
  };
  
  //Keep only k larges coefficients
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

  //Load lena image.
  unsigned int isize = 64;
  cv::Mat dat, img;
  readFITS("../inputs/Lena.fits", dat);
  dat.convertTo(img, cv::DataType<double>::type);
  cv::flip(img, img, -1);
  cv::Rect rect(100, 100, isize, isize); //Take 128x128 image patch at position [100, 100]
  img = img(rect).clone();
  
  //create matrix A mxn with a cosine frequency at every column
  cv::Mat phi;  //psi, matrix that transform in the sparse domain
  cv::Mat eye_nn = cv::Mat::eye(isize, isize, cv::DataType<double>::type);
  cv::idct(eye_nn, phi, cv::DCT_ROWS);
  phi = phi.t();
//  cv::Mat psi;  //sensing matrix. dimension reduction
  unsigned int a = 20;  //number of incoheren measurements
  cv::Mat shuffle_eye = shuffleRows(eye_nn);
  cv::Mat A = shuffle_eye(cv::Range(0, a), cv::Range::all());
  
  cv::Mat img_dct;
  if(false)
  {
    cv::dct(img.row(2), img_dct);
  }
  else
  {
    img_dct = phi.t() * img.row(2).t();
  }
  
  hardThreshold(img_dct, 3);
  std::cout << "img_dct: " << img_dct<< std::endl;
  
  phi = A * phi;
  cv::Mat y;
  if(false)
  {
    cv::idct(img_dct, y);
    y = y.t();
  }
  else
  {
    y = phi * img_dct;
  }
 
  std::cout << "y: " << y << std::endl;
  
  cv::Mat x0 = cv::Mat::zeros(isize, 1, cv::DataType<double>::type);
  double mu(1.0);
  for(unsigned int j =0; j<50; ++j)
  {
    x0 = x0 + ( mu * phi.t() * (  y - (phi * x0) ) );
    hardThreshold(x0, 3);
    std::cout << "x0: " << x0.t() << std::endl;
  }
 
}
