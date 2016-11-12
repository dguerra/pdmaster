/*
 * PhaseScreen.cpp
 *
 *  Created on: Sep 15, 2016
 *      Author: dailos
 */

#include "PhaseScreen.h"
#include "ToolBox.h"

PhaseScreen::PhaseScreen()
{

}

PhaseScreen::~PhaseScreen()
{
  // TODO Auto-generated destructor stub
}


cv::Mat PhaseScreen::fractalMethod(const unsigned int& nw, const double& r0, const double& L0)
{
  /*
  En esa función las unidades de r0 y L0, magnitudes espaciales, están en píxeles. 
  Los valores típicos dependen de la aplicación que estas usando. 
  Valores típicos de r0 para seeing atmosférico bueno son de 200-250mm (0.5-0.4 arcsec fwhm) , 
  para un seeing mediano r0=100mm (1arcsec). L0 es muy difícil de medir en la atmósfera, 
  observaciones en telescopio grandes (8-10m) hace pensar que L0 es de decenas de metros pero como te digo es difícil de medir.
  */
  
  //function w=phasescreen(nw,r0,L0)
  //% w=phasescreen(nw,r0,L0)
  //%   D(r) = 6.88*(r/r0)^(5/3)   r<L0
  //%   D(r) = 6.88*(L0/r0)^(5/3)  r>=l0
  //%   sigma_w = sqrt(1/2 D(L0))
  auto Dw = [](const double& r, const double& r0, const double& L0) -> double { return 6.88 * std::pow( std::min(r,L0)/r0, 5.0/3.0 ); };
  
  auto Cw = [](const double& r, const double& r0, const double& L0) -> double { return 3.44 * std::pow(L0/r0,5.0/3.0) - 3.44 * std::pow(std::min(r,L0)/r0, 5.0/3.0); };
  
  
  unsigned int np = std::ceil(std::log(nw-1) / std::log(2));
  unsigned int n = std::pow(2,np) + 1;
  cv::Mat w = cv::Mat::zeros(n, n, cv::DataType<double>::type);
  double wrms2 = (1.0/2.0) * Dw(L0,r0,L0);
  
  double c0 = wrms2;
  double r = std::min(double(n) - 1.0, L0);
  
  double c1 = wrms2 - (1.0/2.0) * Dw(double(n) - 1.0, r0, L0);
  r = std::min(std::sqrt(2.0) * (double(n) - 1), L0);
  
  double c2 = wrms2 - 1.0/2.0 * Dw(std::sqrt(2.0) * (double(n) - 1.0), r0, L0);
  
  double dataC[] =   {c0, c1, c1, c2, c1, c0, c2, c1, c1, c2, c0, c1, c2, c1, c1, c0};
  cv::Mat C(4, 4, cv::DataType<double>::type, dataC);  // Covariance matrix
  
  cv::Mat K;
  cholesky(C, K);  //C=K*K'
  
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::Mat rnd(4,1,cv::DataType<double>::type);
  cv::randn(rnd, cv::Scalar(0.0), cv::Scalar(1.0));  //cv function randn(dst, mean, stddev) 
  cv::Mat u = K*rnd;
  
  
  
  double* u_d = (double*)u.data;
  double* w_d = (double*)w.data;
  
  w_d[0 + 0 * n] = u_d[0];
  w_d[0 + (n-1) * n] = u_d[1];
  w_d[(n-1) + 0 * n] = u_d[2];
  w_d[(n-1) + (n-1) * n] = u_d[3];

  cv::RNG& rng = cv::theRNG();
  
  for(unsigned int il=0; il<np; ++il)
  { 
    double d = std::pow(2.0,double(np-il));   //cell side
    double alfa = Cw(d/std::sqrt(2.0),r0,L0)/(wrms2 + 2.0 * Cw(d,r0,L0) + Cw(std::sqrt(2.0)*d,r0,L0));
    double alfa0 = std::sqrt(wrms2 - 4.0*std::pow(alfa,2.0)*(wrms2+2.0*Cw(d,r0,L0)+Cw(std::sqrt(2.0)*d,r0,L0)));
    unsigned int d_int(d);
    
    for(unsigned int px=d_int/2; px<n;px=px+d_int)
    {
      for(unsigned int py=d_int/2; py<n;py=py+d_int)
      {
          w_d[py+px*n] = alfa0 * rng.gaussian(1.0) + alfa * ( w_d[(py+d_int/2)+(px+d_int/2)*n] + w_d[(py+d_int/2)+(px-d_int/2)*n] +
                                                              w_d[(py-d_int/2)+(px-d_int/2)*n] + w_d[(py-d_int/2)+(px+d_int/2)*n] );
      }
    }


    if(il>0)
    {
      alfa = Cw(d/2.0,r0,L0)/(wrms2 + 2.0 * Cw(d/std::sqrt(2.0),r0,L0)+Cw(d,r0,L0));
      alfa0 = std::sqrt(wrms2 - 4.0 * std::pow(alfa,2.0) * (wrms2 + 2.0 * Cw(d/std::sqrt(2.0),r0,L0) + Cw(d,r0,L0)));
      
      for(unsigned int q=d_int; q<n-d_int; q=q+d_int)
      {
        for(unsigned int p=d_int/2; p<n;p=p+d_int)
        {
          double randomNumber = 3.5;  //RNG::gaussian(1.0);
          w_d[q+p*n] = alfa0 * rng.gaussian(1.0) + alfa * (w_d[q+(p+d_int/2)*n]+w_d[q+(p-d_int/2)*n]+w_d[(q+d_int/2)+p*n]+w_d[(q-d_int/2)+p*n]);
          w_d[p+q*n] = alfa0 * rng.gaussian(1.0) + alfa * (w_d[(p+d_int/2)+q*n]+w_d[(p-d_int/2)+q*n]+w_d[p+(q+d_int/2)*n]+w_d[p+(q-d_int/2)*n]);
        }
      }
    }

    unsigned int A_dim(2);
    cv::Mat A(A_dim, A_dim, cv::DataType<double>::type);
    double* A_d = (double*)A.data;
    A_d[0 + 0 * A_dim] = wrms2 + Cw(d,r0,L0);
    A_d[1 + 0 * A_dim] = Cw(d/std::sqrt(2.0),r0,L0);
    A_d[0 + 1 * A_dim] = 2.0 * Cw(d/std::sqrt(2.0),r0,L0);
    A_d[1 + 1 * A_dim] = wrms2;
    
    cv::Mat x;   //// Ax = b: x = cv::solve(A, b);     % A\b or mldivide(A,b)
    cv::solve(A, cv::Mat::ones(2,1,cv::DataType<double>::type), x, cv::DECOMP_QR);
    cv::Mat alfa_m = x * Cw(d/2.0,r0,L0);
    double* alfa_m_d = (double*)alfa_m.data;
    alfa0 = std::sqrt(wrms2 - 2.0 * std::pow(alfa_m_d[0],2.0) * (wrms2 + Cw(d,r0,L0)) -
                                    std::pow(alfa_m_d[1],2.0) *  wrms2 - 
                              4.0 * alfa_m_d[0] * alfa_m_d[1] * Cw(d/std::sqrt(2.0),r0,L0) );
    
    
    for(unsigned int p=d_int/2; p<n;p=p+d_int)
    {
      w_d[0 + p*n] = alfa0 * rng.gaussian(1.0) + alfa_m_d[0] * (w_d[0+(p+d_int/2)*n] + w_d[0+(p-d_int/2)*n]) + alfa_m_d[1] * w_d[(d_int/2)+p*n];
      w_d[(n-1) + p*n] = alfa0 * rng.gaussian(1.0) + alfa_m_d[0] * (w_d[(n-1)+(p+d_int/2)*n] + w_d[(n-1)+(p-d_int/2)*n]) + alfa_m_d[1] * w_d[(n-1-d_int/2)+p*n];
      w_d[p + 0*n] = alfa0 * rng.gaussian(1.0) + alfa_m_d[0] * (w_d[(p+d_int/2)+0*n] + w_d[(p-d_int/2)+0*n]) + alfa_m_d[1] * w_d[p+(d_int/2)*n];
      w_d[p + (n-1)*n] = alfa0 * rng.gaussian(1.0) + alfa_m_d[0] * (w_d[(p+d_int/2)+(n-1)*n] + w_d[(p-d_int/2) + (n-1)*n]) + alfa_m_d[1] * w_d[p+(n-1-d_int/2)*n];
    }
  }
  
  return w;
}