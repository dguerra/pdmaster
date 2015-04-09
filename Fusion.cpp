/*
 * Fusion.cpp
 *
 *  Created on: Jan 16, 2015
 *      Author: dailos
 */
#include "Fusion.h"
#include "WaveletTransform.h"
#include "PDTools.h"
#include "FITS.h"
#include "curvelab/fdct_wrapping.hpp"
#include "curvelab/fdct_wrapping_inline.hpp"
using namespace fdct_wrapping_ns;

//committedObject
//meaningfulObject
//imageFusion
//C is equal to A where A is representative and equal to B where A is not representative
//estimate object: F = Wo_1 + Wo_2 + Wo_3 + ... + Wo_N
//original image:  Dj = Wi_1 + Wi_2 + Wi_3 + ... + Wi_N
//Replace every image Dj with the following:
//summation over n = {1-N} of {Wo_n + mask_n * ( Wi_n - Wo_n )}
//where mask_n indicates where information is located


double l1_norm(cv::Mat F)
{
  //fdct_wrapping_
  std::vector< vector<CpxNumMat> > curveletsCoeffs;  //vector<int> extra;
  int nbscales(6), nbangles_coarse(8), ac(1);
  cv::Mat f; //object estimate, measure space
  cv::idft(F, f, cv::DFT_REAL_OUTPUT);
  int m = f.rows, n = f.cols;
  //std::cout << "m: " << m << "n: " << n << std::endl;
  
  CpxNumMat f_NumMat(m,n);
  
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      f_NumMat(i,j) = f.at<double>(j,i);
  //cv::Mat proof(f_NumMat.m(), f_NumMat.n(), cv::DataType<std::complex<double> >::type, f_NumMat.data() );  
  //writeFITS(splitComplex(proof).first, "../curvAbs.fits");
  fdct_wrapping(m, n, nbscales, nbangles_coarse, ac, f_NumMat, curveletsCoeffs);
  double l1_norm(0.0);
  for(auto i = curveletsCoeffs.begin(); i != curveletsCoeffs.end(); ++i)
  { 
    for(auto j = i->begin(); j != i->end(); ++j)
    {
      static int ii(0);
      cv::Mat curv(j->m(),  j->n(), cv::DataType<double>::type, j->data());
      l1_norm += cv::norm(curv, cv::NORM_L1);
    }
  }
  return l1_norm;
}

void fuse(const cv::Mat& A, const cv::Mat& B, const double& sigmaNoise, cv::Mat& fusedImg)
{
  std::vector<cv::Mat> AWavelets, BWavelets;
  cv::Mat AResidual, BResidual;
  unsigned int total_planes(4);
  swtSpectrums_(A, AWavelets, AResidual, total_planes);
  swtSpectrums_(B, BWavelets, BResidual, total_planes);
  
  std::vector<double> wNoiseFactor;

  waveletsNoiseSimulator(sigmaNoise, total_planes, A.size(), wNoiseFactor);
  fusedImg = cv::Mat::zeros(A.size(), A.depth());  //fused image

  for(unsigned int nplane = 0; nplane<total_planes; ++nplane)
  {
    cv::Mat AWj = AWavelets.at(nplane);
    fftShift(AWj);
    cv::idft(AWj, AWj, cv::DFT_REAL_OUTPUT);
    
    cv::Mat BWj = BWavelets.at(nplane);
    fftShift(BWj);
    cv::idft(BWj, BWj, cv::DFT_REAL_OUTPUT);
   
    cv::Mat mask;
    unsigned int windowDim = 2 * (std::pow(2, nplane)+1) + 1;   //5, 7, 11
    //noise == poissonNoise.mul(noiseSigmaWavelet.at(nplane))!!!!!
    cv::Mat noise( AWj.size(), AWj.type(), cv::Scalar(wNoiseFactor.at(nplane)) );
    //mask = cv::Mat::ones(AWj.size(), AWj.type());
    probabilisticMask_(AWj-BWj, noise, cv::Size(windowDim, windowDim), mask);
   
    fusedImg += BWj + mask.mul( AWj - BWj );
  }
  cv::dft(fusedImg, fusedImg, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  fftShift(fusedImg);
  //Add the residual
  fusedImg += AResidual;
}

void waveletsNoiseSimulator(const double& sigmaNoise, const unsigned int& total_planes, const cv::Size& simulationSize, std::vector<double>& wNoiseFactor)
{
  //Propagation of noise through wavelet planes
  //create noise image
  cv::Mat gaussNoise(simulationSize, cv::DataType<double>::type);
  cv::theRNG() = cv::RNG( time (0) );
  cv::randn(gaussNoise, cv::Scalar(0.0), cv::Scalar(sigmaNoise));
  std::vector<cv::Mat> w_noise;
  cv::Mat wr_noise;
  udwd(gaussNoise, w_noise, wr_noise, total_planes);
  cv::Scalar mean, stdDev;
  wNoiseFactor.clear();
  for(cv::Mat nw : w_noise)
  {
    cv::meanStdDev(nw, mean, stdDev);
    wNoiseFactor.push_back(stdDev.val[0]);
  }
}

void swtSpectrums_(const cv::Mat& imgSpectrums, std::vector<cv::Mat>& wavelet_planes, cv::Mat& residu, const unsigned int& total_planes)
{
  cv::Mat source;
  imgSpectrums.copyTo(source);
  
  wavelet_planes.clear();
  //discrete filter derived from scaling function. In our calculation a 1D spline of degree 3
  double m[] = {1.0, 4.0, 6.0, 4.0, 1.0};
  int row_elements = sizeof(m) / sizeof(m[0]);
  cv::Mat row_ref(row_elements, 1, cv::DataType<double>::type, m);
  double scale_factor(2.0);
  cv::Mat scaling_function = row_ref * row_ref.t();
  
  for(unsigned int nplane = 0; nplane < total_planes; ++nplane)
  {
    scaling_function = scaling_function / cv::sum(scaling_function).val[0];
    cv::Mat kernelPadded = cv::Mat::zeros(source.size(), source.depth());
    scaling_function.copyTo(selectCentralROI(kernelPadded, scaling_function.size()));

    cv::Mat kernelPadded_ft;
    cv::dft(kernelPadded, kernelPadded_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
    cv::mulSpectrums(source, kernelPadded_ft.mul(kernelPadded.total()), residu, cv::DFT_COMPLEX_OUTPUT);
 
    wavelet_planes.push_back(source - residu);
    //update variables
    residu.copyTo(source);
    cv::resize(scaling_function, scaling_function, cv::Size(0,0), scale_factor, scale_factor, cv::INTER_NEAREST);
  }  
}

void probabilisticMask_(const cv::Mat& data, const cv::Mat& noise, const cv::Size& windowSize, cv::Mat& mask)
{
//  cv::Mat kernel = cv::Mat::zeros(data.size(), data.type());
//  selectCentralROI(kernel, windowSize) = cv::Scalar(1.0/(windowSize.height * windowSize.width));
  cv::Mat kernel(windowSize, data.type(), cv::Scalar(1.0/(windowSize.height * windowSize.width)));
  cv::Mat dataSigma2, dataSigma, dataSigma_noiseSigma, expS;
  conv_flaw(data.mul(data), kernel, dataSigma2);
  cv::sqrt(dataSigma2, dataSigma);
  double maskFactor(3.0/2.0);
  dataSigma_noiseSigma = maskFactor * (dataSigma - noise);
  dataSigma_noiseSigma.setTo(0, dataSigma_noiseSigma < 0);
  cv::exp((-1) * dataSigma_noiseSigma.mul(dataSigma_noiseSigma)/(2*noise.mul(noise)), expS);
  mask = 1 - expS;
}

