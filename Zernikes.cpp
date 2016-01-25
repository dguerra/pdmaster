//============================================================================
// Name        : Zernikes.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Zernikes.h"
#include "EffectivePixel.h"
#include "PDTools.h"
#include "FITS.h"
#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"
#include "CustomException.h"
#include "TelescopeSettings.h"

using namespace std;
//constexpr double PI_4 = -acos(0.0)/2;

//The default constructor of the Zernikes class
Zernikes::Zernikes()
{
//  setNameMap_();
}

//Constructor with a vector of Zernike coeficients.
Zernikes::Zernikes
(const std::map<unsigned int, double>& zernikeCoefs)
{
  zernikeCoefs_ = zernikeCoefs;
//  setNameMap_();
}

//Returns the wavefront abberation of a given Zernike polynom at a given point.
//Default version with input (j, rho, theta)
double Zernikes::pointZernike(const unsigned int& j, const double& rho, const double& theta)
{
  if(rho < 0) throw CustomException("Rho cannot be negative");
  if(j == 0) throw CustomException("Zernike index cannot be zero");

  try
  {
    double Znm = 0;
    if (rho <= 1)//for rho>1, Znm=0
    {
        //From j, calculate n and m
        unsigned int n;
        int m;
        getNM(j, n, m);   //for j = 4 -> n,m = 2,0
        //std::cout << "n,m: " << n << "," << m << " -> " << "j: " << j << std::endl;
        //Calculate Znm (the Zernike for n and m) including the corresponding
        //  normalization.
        double Nnm = 0.5; //std::sqrt(2.0*(n+1.0)/(1.0+(m==0))); //Normalization Factor
         
        //std::cout << "Nnm: " << Nnm << std::endl;

        //std::cout << "Nnm: " << Nnm << std::endl;
        //Get the radial term
        for (int s=0; s <= 0.5*(n-std::abs(m)); ++s)
        {
          Znm += std::pow(-1.0,s) * factorial(n-s) * std::pow(rho,(n-2*s))
                        /( factorial(s)
                          *factorial(static_cast<int>(0.5*(n+std::abs(m))-s))
                          *factorial(static_cast<int>(0.5*(n-std::abs(m))-s)) );
        }
        //Multiply with the expansion coefficioent (*zernIt),
        //         the normalization factor, and the sinosoidal term
        Znm*=Nnm*((m>=0)*cos(m*theta)-(m<0)*sin(m*theta));   //GTC version

    }
    return Znm;
  }
  catch (std::exception &e)
  {
    throw;
  }
}

cv::Mat Zernikes::phaseMapZernikeSum(const unsigned int& sideLength, const double& radiousLength, const cv::Mat& coeffs)
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
    throw CustomException("Zernikes: coeffs is single column vector of doubles.");
  }
}


cv::Mat Zernikes::zernike_function(const double& lambda, const double& r_c, const int& nph, const int& j, const double& angle)
{
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
  
  auto zernike_mn = [](const int& j,int &m,int &n) -> void // j=1...
  {
    n=0;
    int len=1;
    for(int i=1;len<j;++i) len+=(n=i)+1;
    int dl=n+1-len+j;
    m=2*((dl+(n%2))/2)+!(n%2)-1;
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
  double **z=ft2dim(1,nph,1,nph);
  if(j==1)
    for(int x=1;x<=nph;++x)
      for(int y=1;y<=nph;++y) z[x][y]=1.0;
  else{                                       // j>1
    int m,n;
    zernike_mn(j,m,n);
    double *rc=RC(m,n);
    double **r=ft2dim(1,nph,1,nph);
    double **rs=ft2dim(1,nph,1,nph);
    for(int x=1;x<=nph;++x)                   // s=0
      for(int y=1;y<=nph;++y){
        double rr=std::sqrt((double)std::pow(x-xo, 2.0)+(double)std::pow(y-yo, 2.0))/r_c; 
        rs[x][y]=rr*rr;
        r[x][y]=pow(rr,n);
        z[x][y]=r[x][y]*rc[0];
      }
    rs[xo][yo]=1.0;                           // avoid divide by 0
    for(int s=1;s<=(n-m)/2;++s){
      for(int x=1;x<=nph;++x)
        for(int y=1;y<=nph;++y)
          z[x][y]+=(r[x][y]/=rs[x][y])*rc[s];
      if(!(n-2*s)) z[xo][yo]+=rc[s];          // dividing 0 by 1 will never give 1...
    }
    del_ft2dim(rs,1,nph,1,nph);
    del_ft2dim(r,1,nph,1,nph);
    if(m){                                    // m!=0
      double sf=std::sqrt((double)(2.0*(n+1.0)));   //CHANGE!
      if(j%2)                                 // odd
        for(int x=1;x<=nph;++x)
          for(int y=1;y<=nph;++y) z[x][y]*=sf*std::sin(((double)m)*(std::atan2((double)(y-yo),(double)(x-xo))+angle));
      else                                    // even
        for(int x=1;x<=nph;++x)
          for(int y=1;y<=nph;++y) z[x][y]*=sf*std::cos(((double)m)*(std::atan2((double)(y-yo),(double)(x-xo))+angle));
    }else{                                    // m==0
      double sf=std::sqrt((double)(n+1.0));   //CHANGE!
      for(int x=1;x<=nph;++x)
        for(int y=1;y<=nph;++y) z[x][y]*=sf;
    }
    delete[] rc;
  }
  double sum=0.0,N=0.0,rcs=r_c*r_c,dx=0.5/r_c,dy=0.5/r_c;
  for(int x=1;x<=nph;++x){
    double xl=fabs((double)(x-xo))/r_c-dx,xh=fabs((double)(x-xo))/r_c+dx;
    double xhs=std::pow(xh,2.0);
    for(int y=1;y<=nph;++y){
      double yl=fabs((double)(y-yo))/r_c-dy,yh=fabs((double)(y-yo))/r_c+dy;
      double yhs=std::pow(yh, 2.0);
      double rsl=std::pow(xl,2.0)+std::pow(yl,2.0),rsh=xhs+yhs;
      int ti=(rsl<rcs)+(rsh<rcs);
      if(rsl<=1.0)       // good pixel
      {
        if(rsh<1.0)
        {
          // full pixel
          sum+=std::pow(z[x][y], 2.0);
          N+=1.0;
        }
        else
        {           // partial pixel
          double x2=std::sqrt(std::max(1.0-yhs,(double)0.0));
          double y3=std::sqrt(std::max(1.0-xhs,(double)0.0));
          double f=(xh>yh)?(yh-yl)*(std::min(xh,std::max(xl,x2))-xl)/(4*dx*dy):
                         (xh-xl)*(std::min(yh,std::max(yl,y3))-yl)/(4*dx*dy);
          sum+=f*std::pow(z[x][y],2.0);
          N+=f;
        }
      }
    }
  }
  sum/=N;
  cv::Mat Z(cv::Size(nph, nph), cv::DataType<double>::type);
  for(int x=1;x<=nph;++x)
  {
    for(int y=1;y<=nph;++y) 
    {
      //z[x][y]/=lambda*std::sqrt(sum);
      z[x][y] /= 8.0 * std::sqrt(sum);    //test
      Z.at<double>(x-1, y-1) = z[x][y];
    }
  }
  del_ft2dim(z,1,nph,1,nph);
  return Z;
}

cv::Mat Zernikes::pupilAmplitude(const double& r_c, const int& nph)
{
  // initialise pupil
  //pupil=ft2dim(1,nph,1,nph);
  cv::Mat pupil(nph, nph, cv::DataType<double>::type);
  
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
      if(rsl<=1.0){      // inside pixel
        if(rsh<1.0)      // full pixel
          pupil.at<double>(x-1, y-1)=1.0;
        else{            // partial pixel
          double x2=std::sqrt(max(1.0-yhs,(double)0.0));
          double y3=std::sqrt(max(1.0-xhs,(double)0.0));
          double f=(xh>yh)?(yh-yl)*(std::min(xh,std::max(xl,x2))-xl)/(4*dx*dy):
                           (xh-xl)*(std::min(yh,std::max(yl,y3))-yl)/(4*dx*dy);
          pupil.at<double>(x-1, y-1)=f;
        }
      }else pupil.at<double>(x-1, y-1)=0.0; // outside pixel
    }
  }
 
//
  double area=0.0;
  for(int x=1;x<=nph;++x)
    for(int y=1;y<=nph;++y) area+=pupil.at<double>(x-1, y-1);
  
  return pupil;
}


cv::Mat Zernikes::phaseMapZernike(const unsigned int& j, const unsigned int& sideLength, const double& radiousLength)
{
  double lambda = 4.69600e-7; //1.0;
  cv::Mat zz = zernike_function(lambda, radiousLength, sideLength, j, 0.0);
  cv::Mat pu = pupilAmplitude(radiousLength, sideLength);
  return zz.mul(pu);
  
  /*
  if (j == 0) throw CustomException("Zernikes: j must not be zero");
  //if(radiousLength>sideLength/2) throw CustomException("Zernikes: radious longer that half the side");
  unsigned long nPixels(0);
  cv::Mat thePhaseMap = cv::Mat::zeros(sideLength, sideLength, cv::DataType<double>::type);
  //Loop over all pixels in thePhaseMap. We need to know the coordinates.
  //An explanation of the coordinate conventions used:
  //A 1dim array with dim=4 has pixels with carth.center coordinates 0,1,2,3.
  //The carth. coordinate of the center is at 1.5.
  //The maximum radius, normalized to 1 for a 1dim array is
  //at carth. coord -0.5 and +3.5.

  double center = (sideLength-1.)/2.;

  //Half diagonal of the pixel square
  double edgeThickness = std::sqrt(1/2);

  for (unsigned int xIt = 0; xIt < sideLength; ++xIt)
  {
    for (unsigned int yIt = 0; yIt < sideLength; ++yIt)
    {
      double xCoord = xIt - center;
      double yCoord = yIt - center;
      std::complex<double> pixelCoord(xCoord, yCoord);

      //std::complex<double> theCoord(pixelCoord);
      double pixelArea(1.0);

      //first in case the pixel is out side
      if(std::abs(pixelCoord) <= (radiousLength + edgeThickness))
      {
        if(std::abs(pixelCoord)>=(radiousLength - edgeThickness))
        {//if the pixel lies in the thickness of the circle we take
          //the center of mass of the part of the pixel inside the cicle as the coordinates
          //and its value is proportial to the area inside the circle
          //We only consider pixelSize = 1.0 and radiousLength expressed in pixel units
          EffectivePixel ePixel(pixelCoord, 1.0, radiousLength);
          pixelArea = ePixel.effectiveArea();
          pixelCoord = ePixel.effectiveCoord();
        }

        //else if(std::abs(pixelCoord)<(radiousLength - edgeThickness)) thePhaseMap.at<double>(xIt,yIt) = something;
        thePhaseMap.at<double>(xIt,yIt) = pixelArea * pointZernike(j, std::abs(pixelCoord)/radiousLength, std::arg(pixelCoord));
        nPixels++;
      }
    }
  }

  //Not suere if it is needed to do it
  double rmsZ = std::sqrt(cv::sum(thePhaseMap.mul(thePhaseMap)).val[0]/nPixels);
  cv::divide(thePhaseMap, rmsZ, thePhaseMap);
  return thePhaseMap;
  */
}

std::map<unsigned int, cv::Mat> Zernikes::buildCatalog(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength)
{
  //if(radiousLength <= sideLength/2)
  {
    std::map<unsigned int, cv::Mat> catalog;
    for(unsigned int currentIndex=1; currentIndex <= maximumZernikeIndex; ++currentIndex)
    {
      catalog.insert(std::make_pair(currentIndex,phaseMapZernike(currentIndex, sideLength, radiousLength)));
    }

    return catalog;
  }
  //else
  {
    //throw CustomException("Zernikes: radiousLength larger than half the side of the image.");
  }
}


//Same as buildCatalog, but this is a std::vector of zernike polonomials instead of a std::map
std::vector<cv::Mat> Zernikes::zernikeBase(const unsigned int& maximumZernikeIndex, const unsigned int& sideLength, const double& radiousLength)
{
  std::vector<cv::Mat> base;
  for(unsigned int currentIndex=1; currentIndex <= maximumZernikeIndex; ++currentIndex)
  {
    base.push_back(phaseMapZernike(currentIndex, sideLength, radiousLength));
  }

  return base;
}


///Noll ordering scheme
void Zernikes::getNM(const unsigned int& j, unsigned int& n, int& m)
{
    n=0;
    int len=1;
    for(int i=1;len<j;++i) len+=(n=i)+1;
    int dl=n+1-len+j;
    m=2*((dl+(n%2))/2)+!(n%2)-1;
/*  
  if (j == 0)
    throw CustomException("j must not be zero");
  n = static_cast<unsigned int>(std::ceil((-3.+sqrt(9.+8.*(j-1)))/2.));
  //std::cout << "n: " << n << std::endl;
  //Calculate the number of modes up to this n (excluding modes in this n).
  //The formula is smallerModes = Sum_n(n). This can be derived from the
  //number of modes with a given n, which is n+1.
  unsigned int smallerModes = 0;
  for (unsigned int i = 1; i <= n; ++i)
  {
    smallerModes += i;
  }

  //std::cout << "smallerModes: " << smallerModes << std::endl;
  //Find the position i within this n (i>=1)
  int i = j - smallerModes;
  //trace_.out("getNM: j=%i. n=%i, smallerModes=%i, i=%i",j,n,smallerModes,i);
  //Find the absolute value and sign of m
  if (n%2 == 0) //n is even
  {
    if (i%2 == 0) //i is even
    {
      m = i;
    }
    else
    {
      m = i - 1;
    }
  }
  else        //n is odd
  {
    if (i%2 == 0) //i is even
    {
      m = i - 1;
    }
    else
    {
      m = i;
    }
  }

//  if (j%2 != 0)  ////GTC first solution gives different coefficient order, use i instead of j
  if (i%2 == 0)
  {
    m = -m;
  }

  //debug: Test the result for consistency
  //unsigned int j2=getSingleIndex(n,m);
  //if (j2 != j)
  //  trace_.out("getNM: Error in calculation of n,m: %i != %i\n",j2,j);
*/ 
}


void Zernikes::setCoef(const unsigned int& zernikeIndex, const double& zernikeCoef)
{
  zernikeCoefs_[zernikeIndex] = zernikeCoef;
}


double Zernikes::getCoef(const unsigned int& zernikeIndex)
{
  //throws an exception if key does not exist
  return zernikeCoefs_.at(zernikeIndex);
}
