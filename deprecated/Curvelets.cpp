/*
 * Curvelets.cpp
 *
 *  Created on: March 30, 2015
 *      Author: dailos
 */

#include "Curvelets.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <math.h>

using namespace fdct_wrapping_ns;

Curvelets::Curvelets()
{
  // TODO Auto-generated destructor stub
}


Curvelets::~Curvelets()
{
  // TODO Auto-generated destructor stub
}

void Curvelets::fdct(const cv::Mat& I, std::vector< std::vector<cv::Mat> >& c, const bool& real_coeffs)
{
  cv::Mat complexI;
  if(I.channels() == 1)
  {
    cv::Mat planes[] = {I, cv::Mat::zeros(I.size(), I.type())};
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
  }
  else
  {
    I.copyTo(complexI);  
  }
  
  c.clear();
  int m(complexI.rows ), n(complexI.cols );
  int nbscales = std::floor(std::log2(std::min(m,n)))-3;
  int nbangles_coarse(16), ac(0);

  std::vector< std::vector<fdct_wrapping_ns::CpxNumMat> > c_nmat;  //CurveLab
  fdct_wrapping_ns::CpxNumMat x(m,n);
  memcpy( x._data, complexI.data, m*n*sizeof(std::complex<double>) );
  
  fdct_wrapping_ns::fdct_wrapping(m, n, nbscales, nbangles_coarse, ac, x, c_nmat);
      
  for(auto i = c_nmat.begin(); i != c_nmat.end(); ++i)
  { 
    std::vector<cv::Mat> tmpc;
    for(auto j = i->begin(); j != i->end(); ++j)
    {
      cv::Mat cc(j->m(),  j->n(), cv::DataType<std::complex<double> >::type, j->data());
      tmpc.push_back( cc.clone() );
    }
    c.push_back(tmpc);
  }
  
  if(real_coeffs) c2r(c);
}

double Curvelets::l1_norm(const std::vector< vector<cv::Mat> >& c)
{ 
  //Gives the L1 morm of the curvelets coefficients
  double l1norm(0.0);
  for(auto i = c.begin(); i != c.end(); ++i)
  { 
    for(auto j = i->begin(); j != i->end(); ++j)
    {
      l1norm += cv::norm(*j, cv::NORM_L1);
    }
  }
  return l1norm;
}

void Curvelets::ifdct(std::vector< vector<cv::Mat> >& c, cv::Mat& complexI, int m, int n,  const bool& real_coeffs)
{
  if(real_coeffs) r2c(c);
  
  std::vector< std::vector<fdct_wrapping_ns::CpxNumMat> > c_nmat;
  for(auto i = c.begin(); i != c.end(); ++i)
  { 
    std::vector<fdct_wrapping_ns::CpxNumMat> v_tmpc;
    for(auto j = i->begin(); j != i->end(); ++j)
    {
      fdct_wrapping_ns::CpxNumMat tmpc(j->rows, j->cols);
 
      memcpy( tmpc._data, j->data, j->rows*j->cols*sizeof(std::complex<double>) );
      v_tmpc.push_back(tmpc);
    }
    c_nmat.push_back(v_tmpc);
  }
 
  int nbscales = std::floor(std::log2(std::min(m,n)))-3;
  int nbangles_coarse(16), ac(0);
  fdct_wrapping_ns::CpxNumMat y(m,n); fdct_wrapping_ns::clear(y);
  fdct_wrapping_ns::ifdct_wrapping(m, n, nbscales, nbangles_coarse, ac, c_nmat, y);
  complexI = cv::Mat(y.m(),  y.n(), cv::DataType<std::complex<double> >::type, y.data()).clone();
}


void Curvelets::c2r(std::vector< std::vector<cv::Mat> >& c)
{
  // fdct_usfft_c2r - transform complex curvelet coefficients to real coefficients
  cv::Mat planes[2];
  cv::split(c.at(0).at(0), planes);
  planes[0].copyTo( c.at(0).at(0) );
  std::size_t nbs = c.size();
  for(unsigned int s=1; s<nbs; ++s)
  {
    std::size_t nw = c.at(s).size();
    for(unsigned int w=0; w<nw/2; ++w)
    {
      cv::Mat A_planes[2];
      cv::split(c.at(s).at(w), A_planes);
      c.at(s).at(w) = std::sqrt(2.0) * A_planes[0];   //B = C{s}{w+nw/2};
      c.at(s).at(w+(nw/2)) = std::sqrt(2.0) * A_planes[1];
    }
  }
  cv::split(c.at(nbs-1).at(0), planes);
  c.at(nbs-1).at(0) = planes[0];
  
  /*
  function C = fdct_wrapping_c2r(C)
  //fdct_usfft_c2r - transform complex curvelet coefficients to real coefficients
  nbs = length(C);
  C{1}{1} = real(C{1}{1});
  for s=2:nbs
    nw = length(C{s});
    for w=1:nw/2
      A = C{s}{w};      %B = C{s}{w+nw/2};
      C{s}{w} = sqrt(2) * real(A);      C{s}{w+nw/2} = sqrt(2) * imag(A);
    end
  end
  C{nbs}{1} = real(C{nbs}{1});
  */
}

void Curvelets::r2c(std::vector< std::vector<cv::Mat> >& c)
{
  cv::Mat planes[2] = {c.at(0).at(0), cv::Mat::zeros(c.at(0).at(0).size(), c.at(0).at(0).type())};
  cv::merge(planes, 2, c.at(0).at(0));
  // fdct_usfft_r2c - transform real curvelet coefficients to complex coefficients
  std::size_t nbs = c.size();
  for(unsigned int s = 1; s<nbs; ++s)
  {
    std::size_t nw = c.at(s).size();
    for(unsigned int w = 0; w < nw/2; ++w)
    {
      cv::Mat A = c.at(s).at(w);
      cv::Mat B = c.at(s).at(w+(nw/2));
      cv::Mat planes[] = {A, B};
      cv::Mat AB;
      cv::merge(planes, 2, AB);
      c.at(s).at(w) = (1.0/std::sqrt(2.0)) * AB;
      cv::Mat planes_conj[] = {A, -1*B};
      cv::Mat ABconj;
      cv::merge(planes_conj, 2, ABconj);
      c.at(s).at(w+(nw/2)) = (1.0/std::sqrt(2.0)) * ABconj;
    }
  }
  planes[0] = c.at(nbs-1).at(0);
  planes[1] = cv::Mat::zeros(c.at(nbs-1).at(0).size(), c.at(nbs-1).at(0).type());
  cv::merge(planes, 2, c.at(nbs-1).at(0));
  /*
   function C = fdct_wrapping_r2c(C)
% fdct_usfft_r2c - transform real curvelet coefficients to complex coefficients
  nbs = length(C);
  for s=2:nbs
    nw = length(C{s});
    for w=1:nw/2
      A = C{s}{w};    B = C{s}{w+nw/2};
      C{s}{w} = 1/sqrt(2) * (A+i*B);    C{s}{w+nw/2} = 1/sqrt(2) * (A-i*B);
    end
  end
  */
}
