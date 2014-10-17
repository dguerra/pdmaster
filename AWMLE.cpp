/*
 * AWMLE.cpp
 *
 *  Created on: Apr 15, 2014
 *      Author: dailos
 */


#include <iostream>
#include <cmath>
#include <vector>
#include "CustomException.h"
#include "AWMLE.h"
#include "PDTools.h"
#include "WaveletTransform.h"
#include "FITS.h"
#include "PDTools.h"
#include "ImageQualityMetric.h"

//Adaptive wavelet maximum likelihood estimator

void AWMLE(const cv::Mat& img, const cv::Mat& psf, cv::Mat& object, const double& sigmaNoise, const unsigned int& total_planes)
{
  std::vector<cv::Mat> w_img, w_fimg, w_projection;
  cv::Mat wr_img, wr_fimg, wr_projection;
  std::vector<double> noiseSigmaWavelet;
  //simulate noise propagation through wavelets
  waveletNoise(sigmaNoise, total_planes, img.size(), noiseSigmaWavelet);

  //split up image into its wavelet planes
  udwd(img, w_img, wr_img, total_planes);

  //Estimation of the Poisson noise
  cv::Mat poissonNoise;
  cv::sqrt(wr_img + std::pow(sigmaNoise,2), poissonNoise);
  cv::Mat projection, psf_normalized;
  //make sure psf is normalized, and the total engery is one
  cv::divide(psf, cv::sum(psf), psf_normalized);
  std::cout << "psf.normalized.at(40,40): " << psf_normalized.at<double>(40,40) << std::endl;
  object = cv::Mat::ones(img.size(), img.type());   //object initialization
  double likelihood(0.0);
  double convergence = std::numeric_limits<float>::epsilon();  //difference between one float and the following
  double fx_new = std::numeric_limits<float>::max()/10.0;  //A very big number
  double fx_old(0.0);
  cv::Scalar energy = cv::sum(img);
  std::cout << "energy=" << energy.val[0] << std::endl;
  for(unsigned int i(0); i<100; ++i)
  {
    conv_flaw(object, psf_normalized, projection);
    cv::Mat fimg; //filtered image
    calculpprima(img, projection, sigmaNoise, fimg, likelihood);  //filtered image

    fx_old = fx_new;
    fx_new = likelihood;

    udwd(fimg, w_fimg, wr_fimg, total_planes);   //filtered image wavelet decomposition
    udwd(projection, w_projection, wr_projection, total_planes);    //projection wavelet decomposition

    cv::Mat correction = cv::Mat::zeros(fimg.size(), fimg.type());
    cv::Mat mask;
    //for every wavelet plane:
    for(unsigned int nplane = 0; nplane < total_planes; ++nplane )
    {
      unsigned int windowDim = 2*(std::pow(2,nplane)+1)+1;   //5, 7, 11,
      probabilisticMask( w_fimg.at(nplane) - w_projection.at(nplane), poissonNoise.mul(noiseSigmaWavelet.at(nplane)), cv::Size(windowDim, windowDim), mask);
      correction += w_projection.at(nplane) + mask.mul( w_fimg.at(nplane) - w_projection.at(nplane) );
    }

    correction += wr_fimg;

    cv::divide(correction, projection, correction);
    bool conjugatePSF(true);
    cv::Mat o_term;
    conv_flaw(correction, psf_normalized, o_term, conjugatePSF);  //inverse projection
    cv::multiply(object, o_term, object);
    cv::multiply(object, energy/cv::sum(object), object);  // Energy correction
    std::cout << "iteration: " << i << ": likelihood=" <<likelihood << std::endl;
    std::cout << "Energy ratio: " << energy.val[0]/cv::sum(object).val[0] << "; Convergence: " << std::abs((fx_new - fx_old)/((fx_new + fx_old)/2)) << std::endl;
    if(std::abs((fx_new - fx_old)/((fx_new + fx_old)/2)) <=  convergence)
    {
      std::cout << "AWMLE: Convergence limit has been reached." << std::endl;
      break;
    }
  }
}

void probabilisticMask(const cv::Mat& data, const cv::Mat& noise, const cv::Size& windowSize, cv::Mat& mask)
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

void waveletNoise(const double& sigmaNoise, const unsigned int& total_planes, const cv::Size& simulationSize, std::vector<double>& wNoiseFactor)
{
  //Propagation of noise through wavelet planes
  //create noise image
  cv::Mat gaussNoise(simulationSize, CV_64F);
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

void calculpprima(const cv::Mat& img, const cv::Mat& prj, const double& sigmaNoise, cv::Mat& modifiedimage, double& likelihood)
{
  double pi = 2.0 * acos(0.0);
  double pi2 = 2.0 * pi;
  double v = sigmaNoise;
  double vv = v * v;
  double vv2 = 2.0 * vv;

  double exp2 = exp(-1.0/vv);
  double expantor = exp(1.0/vv2);
  double lc = std::log(std::sqrt(pi2*v));
  double maxim = 2.5 * v;
  likelihood = 0.0;
  double new1 = 0.0;
  modifiedimage = cv::Mat::zeros(img.size(), img.type());
  if (sigmaNoise < 1.0E-6) //If sigma is very low, i.e., if noise can be assumed as pure Poisson.
  {
    modifiedimage = img.clone();
//    index = where( imag ne 0., count, COMPLEMENT = no_index, NCOMPLEMENT = ncount)

    cv::Mat logImg, logPrj;
    cv::log(img, logImg);
    cv::log(prj, logPrj);
    likelihood = cv::sum((img.mul(logPrj)-prj) - (img.mul(logImg)-img)).val[0];
  }
  else // if Gaussian noise is not negligible.
  {
    //for each pixel of image and projection
    auto itModifiedimage = modifiedimage.begin<double>();
    for(auto itImg = img.begin<double>(), itPrj = prj.begin<double>(),
             itImgEnd = img.end<double>(), itPrjEnd = prj.end<double>();
             itImg != itImgEnd, itPrj != itPrjEnd; ++itImg, ++itPrj, ++itModifiedimage)
    {
      double p = *itImg;      //image pixel value
      double hh = *itPrj;     //projection pixel value

      long long lower = (long long)(p) - 2.5 * v;
      long long upper = (long long)(p) + 2.5 * v;
      double hp = (hh-p) * (hh-p) + 1.0;
      double lhp = std::log(hp);
      double serieantden, serieantnum;

      if(p <= maxim)
      {
        lower = 0;
        if(upper < 1) upper = 1;
        serieantden = std::exp(-(1.0-p)*(1.0-p)/vv2) * hh/hp;
        serieantnum = serieantden;
        double termeantnum = serieantden;
        double termeantden = serieantden;
        double expant = exp(-(1.0-2.0*p)/vv2);
        for(unsigned int k=2;k<=upper;++k)
        {
          expant = expant * exp2;
          termeantnum = termeantnum * hh/(k-1) * expant;
          termeantden = termeantden * hh/k * expant;
          serieantnum += termeantnum;
          serieantden += termeantden;
        }

        serieantden += exp(-p*p/vv2)/hp;        // adding k=0 term
        new1 = -lc - hh + log(serieantden) + lhp;
      }
      else
      {
        double lp = log(p);
        double lh = log(hh);
        double l2 = log(sqrt(pi2*p));
        serieantden = 1.0/hp;
        serieantnum = p/hp;
        double termeantnumsup = serieantnum;
        double termeantdensup = serieantden;
        double termeantnuminf = serieantnum;
        double termeantdeninf = serieantden;
        double expant = expantor;
        long long k = (long long)(p) + 1;
        long long l = (long long)(p) - 1;

        while((k <= upper) && (l >= lower))
        {
          k += 1;
          l -= 1;
          expant = expant * exp2;

          termeantnumsup = termeantnumsup * hh/(k-1) * expant;
          termeantdensup = termeantdensup * hh/k * expant;
          termeantnuminf = termeantnuminf * l/hh * expant;
          termeantdeninf = termeantdeninf * (l+1)/hh * expant;

          serieantnum += termeantnumsup+termeantnuminf;
          serieantden += termeantdensup+termeantdeninf;
        }

        new1 = log(serieantden) + (p*lh) - l2 - (p*lp) + p + lhp - lc - hh;

      }

      *itModifiedimage = serieantnum / serieantden;

      if(std::isfinite(new1)) likelihood += new1;
    }
  }
}

void perElementFiltering(const cv::Mat& img, const cv::Mat& prj, cv::Mat& out, const double& sigmaNoise)
{
  if(img.size() == prj.size())
  {
    out = cv::Mat::zeros(img.size(), img.type());
    //compute pixel by pixel
    auto itOut = out.begin<double>();

    for(auto itImg = img.begin<double>(), itImgEnd = img.end<double>(),
             itPrj = prj.begin<double>(), itPrjEnd = prj.end<double>();
             itImg != itImgEnd, itPrj != itPrjEnd; ++itImg, ++itPrj, ++itOut )
    {
      double num(0.0), den(0.0);
      for(unsigned int k(0);k<100;++k)
      {
        double A = -std::pow(k-(*itImg),2)/(2*std::pow(sigmaNoise,2));
        double B = std::exp(A) * (std::pow((*itPrj),k)/factorial(k));
        num += k * B;
        den += B;
      }
      *itOut = num/den;
    }
  }
  else
  {
    throw CustomException("AWMLE: Image and projection must have same size.");
  }
}
