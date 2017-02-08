//============================================================================
// Name        : Zernike.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Zernike.h"
#include "ToolBox.h"
#include "FITS.h"
#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"
#include "CustomException.h"
#include "OpticalSetup.h"

//Rename as Representation, Rprsnttn, DictionaryBase, Zernike

using namespace std;
const double PI = 3.141592653589793238462643383279502884197169399375105;

Zernike::Zernike(const double& r_c, const int& nph, const int& z_max) :
  radious_px_(r_c), side_px_(nph), base_(z_max)
{
  cv::Mat c_mask;
  circular_mask(radious_px_, side_px_, c_mask);
  cv::Scalar c_area = cv::sum(c_mask);
  //cv::divide(c_mask, c_area, c_mask);
  for(unsigned int j=0;j<z_max;++j)
  {
    if(false)
    {
      polynomial(radious_px_, side_px_, j + 1, base_.at(j));
      cv::multiply(base_.at(j), c_mask, base_.at(j));
      //Be aware that RMS value should be normalized to one!!
      double l2 = cv::norm(base_.at(j), cv::NORM_L2);
      cv::multiply(base_.at(j), l2/std::sqrt(c_area.val[0]), base_.at(j));
    }
    else
    {
      base_.at(j) = phaseMapZernike(j + 1, side_px_, radious_px_).clone();
    }
  }
}

//The default constructor of the Zernike class
Zernike::Zernike()
{
}

//analysis operator: action analyse; takes the phase signal and gives the sequence of coefficients (representation realm)
//from phase signal space to zernike representation
void Zernike::analyse(const cv::Mat& sig, cv::Mat& z_coeffs)
{
  z_coeffs.release();
  for(cv::Mat z_j : base_)
  {
    double l2 = cv::norm(z_j, cv::NORM_L2);
    double inner_prod = z_j.dot(sig)/(l2*l2);
    z_coeffs.push_back( inner_prod );
  }
}

void Zernike::synthesize(const cv::Mat& z_coeffs, cv::Mat& sig)
{
  if(z_coeffs.cols != 1) throw CustomException("Coeffs must be single column matrix.");
  if(z_coeffs.rows > base_.size()) throw CustomException("Number of coeffs greater than zernike base.");
  sig = cv::Mat::zeros(side_px_, side_px_, cv::DataType<double>::type);
  auto coeffs_it = z_coeffs.begin<double>();
  for(cv::Mat z_j : base_)
  {
     if(coeffs_it == z_coeffs.end<double>()) break;
     if(*coeffs_it != 0.0) sig += (*coeffs_it) * z_j;
     coeffs_it++;
  }
}

//synthesis operator: action synthesize; takes the sequence of coefficients (representation realm) and gives the phase signal
//Rename as zernike_synthesis: from zernike representation to phase signal space
cv::Mat Zernike::phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs)
{
  if(coeffs.cols == 1 && coeffs.type() == cv::DataType<double>::type)
  {
    cv::Mat thePhaseMapSum = cv::Mat::zeros(sideLength, sideLength, cv::DataType<double>::type);

    for(auto cIt = coeffs.begin<double>(), cEnd = coeffs.end<double>(); cIt != cEnd; ++cIt)
    {
      if((*cIt) != 0.0)
      {
        //cv::accumulate((*cIt) * phaseMapZernike(std::distance(coeffs.begin<double>(), cIt) + 1, sideLength, radiousLength), thePhaseMapSum);
        thePhaseMapSum += (*cIt) * phaseMapZernike(std::distance(coeffs.begin<double>(), cIt) + 1, sideLength, radiousLength);
      }
    }

    return thePhaseMapSum;
  }
  else
  {
    throw CustomException("Zernike: coeffs is single column vector of doubles.");
  }
}

void Zernike::zernike_mn(const int& j,int &m,int &n) // j=1...
{
  n = 0;
  int len = 1;
  for(int i = 1;len<j;++i) len += (n=i)+1;
  int dl = n + 1 - len + j;
  m = 2 * ((dl+(n%2))/2)+!(n%2)-1;
}

void Zernike::polynomial(const double& r_c, const int& nph, const int& j, cv::Mat& z_j)
{
  double angle = 0.0;   //The angle value is set to zero by default. Use it when needed
  auto ft2dim = [](const int& x1l, const int& x1h, const int& x2l, const int& x2h) -> double **
  {
    int nx1=x1h-x1l+1,nx2=x2h-x2l+1;
    double **p;
    p=new double* [nx1] - x1l;
    p[x1l] = new double [nx1*nx2] - x2l;
    for(int x1=x1l+1;x1<=x1h;++x1) p[x1]=p[x1-1]+nx2;
    return p;
  };

  auto del_ft2dim = [](double **p,const int& x1l, const int& x1h, const int& x2l, const int& x2h)-> void
  {
    delete[] (p[x1l]+x2l);
    delete[] (p+x1l);
  };
  
  auto RC = [](const int& m, const int& n) -> double *
  {
    int nmm=(n-m)/2,npm=(n+m)/2;
    int nmax=std::max(npm,n);
    double *f=new double [nmax+1];
    f[0]=1.0;
    for(int i=1;i<=nmax;++i) f[i]=(double)i*f[i-1];
    double *res=new double [nmm+1]; 
    for(int s=0,pm=-1;s<=nmm;++s)
      res[s]=(double)((pm*=-1)*f[n-s])/(f[s]*f[npm-s]*f[nmm-s]);
    delete[] f;
    return res;
  };
  
  int xo=1+nph/2,yo=1+nph/2;
  
  //Release array if necesary
  z_j.create(nph, nph, cv::DataType<double>::type);
  double* z = (double*)z_j.data;

  
  if(j==1)
  {
    for(int x=1;x<=nph;++x)
      for(int y=1;y<=nph;++y) z[(x-1)*nph + (y-1)] = 1.0;
  }
  else
  {                                       // j>1
    int m,n;
    zernike_mn(j,m,n);
    double *rc=RC(m,n);
    double **r=ft2dim(1,nph,1,nph);
    double **rs=ft2dim(1,nph,1,nph);
    for(int x=1;x<=nph;++x)                   // s=0
    {
      for(int y=1;y<=nph;++y)
      {
        double rr=std::sqrt((double)std::pow(x-xo, 2.0)+(double)std::pow(y-yo, 2.0))/r_c; 
        rs[x][y]=rr*rr;
        r[x][y]=pow(rr,n);
        z[(x-1)*nph + (y-1)]=r[x][y]*rc[0];
      }
    }
    rs[xo][yo]=1.0;                           // avoid divide by 0
    for(int s=1;s<=(n-m)/2;++s)
    {
      for(int x=1;x<=nph;++x)
      {
        for(int y=1;y<=nph;++y)
        {  
          z[(x-1)*nph + (y-1)]+=(r[x][y]/=rs[x][y])*rc[s];
        }
      }  
      if(!(n-2*s)) z[(xo-1)*nph + (yo-1)]+=rc[s];          // dividing 0 by 1 will never give 1...
    }
    del_ft2dim(rs,1,nph,1,nph);
    del_ft2dim(r,1,nph,1,nph);
    if(m)
    {                                    // m!=0
      double sf=std::sqrt((double)(2.0*(n+1.0)));   //CHANGE!
      if(j%2)
      {   // odd
        for(int x=1;x<=nph;++x)
        {
          for(int y=1;y<=nph;++y) z[(x-1)*nph + (y-1)]*=sf*std::sin(((double)m)*(std::atan2((double)(y-yo),(double)(x-xo))+angle));
        }
      }
      else                                    // even
      {
        for(int x=1;x<=nph;++x)
        {
          for(int y=1;y<=nph;++y) z[(x-1)*nph + (y-1)]*=sf*std::cos(((double)m)*(std::atan2((double)(y-yo),(double)(x-xo))+angle));
        }
      }
    }
    else
    {                                    // m==0
      double sf=std::sqrt((double)(n+1.0));   //CHANGE!
      for(int x=1;x<=nph;++x)
      {
        for(int y=1;y<=nph;++y) z[(x-1)*nph + (y-1)]*=sf;
      }
    }
    delete[] rc;
  }
  
//  return cv::Mat(cv::Size(nph, nph), cv::DataType<double>::type, &z[1][1]);  //c-like array starts at position (1,1)
}

double Zernike::zernike_covar(int i,int j)
{
  auto gammln = [](double xx, double &sign)-> double
  {
    static double cof[6]={ 7.618009172947146E+01,-8.650532032941677E+01, 2.401409824083091E+01,
                           -1.231739572450155E+00, 1.208650973866179E-03,-5.395239384953000E-06};
  /*
    double x=xx;
    double y=x;
    double tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    double ser=1.000000000190015;
    for (int j=0;j<=5;++j) ser+=cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
  */
    double yy = xx;
    double res = 1.000000000190015;
    while(yy < 1.0)
    {
      res *= yy;
      yy += 1.0;
    }
    
    sign = (res >= 0) ? 1.0 : -1.0;
    yy -= 1.0;
    double tmp = yy + 5.5;
    tmp -= (yy + 0.5) * std::log(tmp);
    double ser = 1.000000000190015;
    for (int jj = 0; jj <= 5; ++jj) ser += cof[jj] / ++yy;
    return -tmp + std::log(2.5066282746310005 * ser) - std::log(std::fabs(res));
  };


  if((i<2)||(j<2)) return 0.0;
  int m,n,o,p;
  zernike_mn(i,m,n);
  zernike_mn(j,o,p);
  if(m!=o) return 0.0;
  if(m) if((i+j)%2) return 0.0;
//  ; Now deal with the numerical terms: Dai
  double tmp;
  double k = std::pow(4.8 * std::exp(gammln(6.0/5.0,tmp)),5.0/6.0) * std::exp(gammln(14.0/3.0,tmp)+2.0*gammln(11.0/6.0,tmp))/(std::pow(2.0,(8.0/3.0))*PI);
  k *= std::pow(-1.0,(double)((n+p-2*m)/2)) * std::sqrt((double)((n+1)*(p+1)));
  double g1_sgn,g2_sgn,g3_sgn,g4_sgn;
  double g1 = gammln(((double)(n+p)- 5.0/3.0)/2.0,g1_sgn);
  double g2 = gammln(((double)(n-p)+17.0/3.0)/2.0,g2_sgn);
  double g3 = gammln(((double)(p-n)+17.0/3.0)/2.0,g3_sgn);
  double g4 = gammln(((double)(n+p)+23.0/3.0)/2.0,g4_sgn);
  return k * std::exp(g1 - g2 - g3 - g4) * g1_sgn * g2_sgn * g3_sgn * g4_sgn;
}

void Zernike::circular_mask(const double& r_c, const int& nph, cv::Mat& c_mask)
{
  c_mask.create(nph, nph, cv::DataType<double>::type);
  double* z = (double*)c_mask.data;

  //Circular mask: zero-valued outside the cicle, one-valued inside, 
  //the edge pixels has values between 0 and 1 depending on how far lies the center of the pixel from the edge
  int xo=1+nph/2,yo=1+nph/2;
  double rcs=r_c*r_c,dx=0.5/r_c,dy=0.5/r_c;
  for(int x=1;x<=nph;++x){
    double xl=fabs((double)(x-xo))/r_c-dx,xh=fabs((double)(x-xo))/r_c+dx;
    double xhs=std::pow(xh,2.0);
    for(int y=1;y<=nph;++y){
      double yl=fabs((double)(y-yo))/r_c-dy,yh=fabs((double)(y-yo))/r_c+dy;
      double yhs=std::pow(yh,2.0);
      double rsl=std::pow(xl,2.0)+std::pow(yl,2.0),rsh=xhs+yhs;
      int ti=(rsl<rcs)+(rsh<rcs);
      if(rsl<=1.0)
      {      // inside pixel
        if(rsh<1.0)      // full pixel
        {
          z[(x-1)*nph + (y-1)] = 1.0;
        }
        else
        {            // partial pixel
          double x2 = std::sqrt(max(1.0-yhs,(double)0.0));
          double y3 = std::sqrt(max(1.0-xhs,(double)0.0));
          double f = (xh>yh)?(yh-yl)*(std::min(xh,std::max(xl,x2))-xl)/(4*dx*dy):
                           (xh-xl)*(std::min(yh,std::max(yl,y3))-yl)/(4*dx*dy);
          z[(x-1)*nph + (y-1)] = f;
        }
      }
      else z[(x-1)*nph + (y-1)] = 0.0; // outside pixel
    }
  }
 
}

void Zernike::circular_mask(cv::Mat& c_mask)
{
  double r_c(radious_px_);
  int nph(side_px_);
  c_mask.create(nph, nph, cv::DataType<double>::type);
  double* z = (double*)c_mask.data;

  //Circular mask: zero-valued outside the cicle, one-valued inside, 
  //the edge pixels has values between 0 and 1 depending on how far lies the center of the pixel from the edge
  int xo=1+nph/2,yo=1+nph/2;
  double rcs=r_c*r_c,dx=0.5/r_c,dy=0.5/r_c;
  for(int x=1;x<=nph;++x){
    double xl=fabs((double)(x-xo))/r_c-dx,xh=fabs((double)(x-xo))/r_c+dx;
    double xhs=std::pow(xh,2.0);
    for(int y=1;y<=nph;++y){
      double yl=fabs((double)(y-yo))/r_c-dy,yh=fabs((double)(y-yo))/r_c+dy;
      double yhs=std::pow(yh,2.0);
      double rsl=std::pow(xl,2.0)+std::pow(yl,2.0),rsh=xhs+yhs;
      int ti=(rsl<rcs)+(rsh<rcs);
      if(rsl<=1.0)
      {      // inside pixel
        if(rsh<1.0)      // full pixel
        {
          z[(x-1)*nph + (y-1)] = 1.0;
        }
        else
        {            // partial pixel
          double x2 = std::sqrt(max(1.0-yhs,(double)0.0));
          double y3 = std::sqrt(max(1.0-xhs,(double)0.0));
          double f = (xh>yh)?(yh-yl)*(std::min(xh,std::max(xl,x2))-xl)/(4*dx*dy):
                           (xh-xl)*(std::min(yh,std::max(yl,y3))-yl)/(4*dx*dy);
          z[(x-1)*nph + (y-1)] = f;
        }
      }
      else z[(x-1)*nph + (y-1)] = 0.0; // outside pixel
    }
  }
 
}


cv::Mat Zernike::phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength, const bool& unit_rms, const bool& c_mask)
{
  //Find the rms for this mode:
  auto rms_t = [](const cv::Mat& zf, const cv::Mat& mask) -> double
  {
    cv::Mat zf2, cmzf2;
    cv::multiply(zf, zf, zf2);
    cv::multiply(mask, zf2, cmzf2);
    cv::Scalar rms2 = cv::sum(cmzf2) / cv::sum(mask);
    return std::sqrt(rms2.val[0]);
  };
  
  cv::Mat zz;
  polynomial(radiousLength, sideLength, j, zz);
  cv::Mat pu;
  circular_mask(radiousLength, sideLength, pu);
  cv::Mat zernike_mode;
  
  if(c_mask) cv::multiply(zz, pu, zernike_mode);
  else zernike_mode = zz.clone();
  
  double rms_d = rms(zz, pu);
  if(rms_d != rms_t(zz,pu)) throw CustomException("Rms problem in phaseMapZernike.");
  if(unit_rms) zernike_mode = zernike_mode/rms_d;
  return zernike_mode/10.0;    //divide by 10.0 in order to make sparse recovery works
}
