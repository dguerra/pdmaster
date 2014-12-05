/*
 * Minimization.h
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#ifndef MINIMIZATION_H_
#define MINIMIZATION_H_

#include <iostream>
#include <vector>
#include <limits>  //numeric_limit<double>::epsilon
#include <algorithm>    // std::max
#include <cmath>   //std::abs
#include <functional>   //function objects
#include "opencv2/opencv.hpp"
#include "CustomException.h"


//The process of finding the set of alpha values that gives the optimum phase aberration is non-linear
//here we aim to achieve a way to find the optimum increment in alpha values to be nearer to optimum solution in the next iteration
//The system described above IS linear and the new variable (x) is actually delta of alpha, which gives direction and magnitude towards the minimum
//and can be written as Ax-b=0, being dim(A)={MxM} and dim(b)={1xM}, and M the number of parameter we chose to define the aberration

//The result are two matrix, A and b that summarizes the whole linear equation system
//A matrix could be seen as Hessian matrix and b as the gradient vector
//enum class MinimizationMethod {Newton, BFGS , ConjugateGradients};
//

//Helpter struct that turns a multidimensional functor into 1-dim, through point p and direction xi
template<class T>
struct F1dim
{  //could be implemented through function adaptors, read more about it
  cv::Mat_<double> p_;
  cv::Mat_<double> xi_;
  T func_;
    
  F1dim(const cv::Mat_<double> &pp, const cv::Mat_<double> &xii, T &funcc)
  {
    pp.copyTo(p_); xii.copyTo(xi_); func_ = funcc;
  }
  double operator() (const double x)
  {
   return func_(p_ + x * xi_);
  }
};

class Minimization
{
public:
  Minimization(){};
  virtual ~Minimization(){};
  
  template <class T>
  void bracket(const double& a, const double& b, T &func);
  double ax, bx, cx, fa, fb, fc;   //Variable used by bracket method
  
  template <class T> double brent(T &func);
  double xmin,fmin;   //Variable used by brent 1D minimization method
	const double tol = 3.0e-8;   //Precision at which the minimum is found
  const double gtol = 3.0e-8;   //Precision gradient at which the minimum is found
  
  const int ITMAX = 200;  //maximum number of steps to reach the minimum
  //Search func minimum along xi direction from point p
  template<class T>
  double linmin(cv::Mat_<double>& p, cv::Mat_<double>& xi, T &func);
  
  //Build gradient and set next point a direction in convergence to the minimum
  template <class T, class U>
  void dfpmin(cv::Mat_<double> &p, int &iter, double &fret, T &func, U &dfunc);
  
  template <class T, class U>
  int nextStep(cv::Mat_<double> &p, cv::Mat_<double> &xi, cv::Mat_<double> &g, 
               cv::Mat_<double> &hessin, double &fret, T &func, U &dfunc);
 
  template<class T, class U>
  void minimize(cv::Mat_<double> &p, const cv::Mat_<double> &Q2_constraints, int &iter, double &fret, 
                T &func, U &dfunc);
};

template<class T, class U>
void Minimization::minimize(cv::Mat_<double> &p, const cv::Mat_<double> &Q2_constraints, int &iter, double &fret, 
                            T &func, U &dfunc)
{
  //build up new functions func and dfunc in the constrained space with fewer unknowns
//  F_constrained f_constrained(func, Q2_constraints);
//  DF_constrained df_constrained(dfunc, Q2_constraints);
  //Starting point in the constraints space
//  cv::Mat_<double> p_constrained = cv::Mat::zeros(Q2_constraints.cols, 1, cv::DataType<double>::type);
//  dfpmin(p_constrained, iter, fret, f_constrained, df_constrained);
//  p = Q2_constraints * p_constrained;
}

template <class T, class U>
void Minimization::dfpmin(cv::Mat_<double> &p, int &iter, double &fret, T &func, U &dfunc)
{
	int n = p.total();   //Check the vector has only one column first
 
	cv::Mat_<double> g, xi;
	cv::Mat_<double> hessin = cv::Mat::eye(n, n, cv::DataType<double>::type);  //initialize to identity matrix
	fret = func(p);
	g = dfunc(p);
  xi = -1 * g;  //first direction is the opposite to the gradient
  
  
  //variables: p:[point], xi:[search direction], func:[function], dfunc:[gradient function], g:[gradient at p], h:[hessian at p], 
  
	for (int its=0;its<ITMAX;its++)
  {
		iter = its;
    if(nextStep(p, xi, g, hessin, fret, func, dfunc)) return;   //minimum reached
	}
	throw("too many iterations in dfpmin");
}

template <class T, class U>
int Minimization::nextStep(cv::Mat_<double> &p, cv::Mat_<double> &xi, cv::Mat_<double> &g, 
                           cv::Mat_<double> &hessin, double &fret, T &func, U &dfunc)
{
	const double EPS = std::numeric_limits<double>::epsilon();
	const double TOLX = 4 * EPS;
  double den, fac, fad, fae, sumdg, sumxi;
  cv::Mat_<double> dg, hdg;
  //we use linmin uses brent method inside to look for the minimum in 1D
  fret = linmin(p, xi, func);
				
  cv::Mat_<double> temp;
	cv::Mat_<double> abs_p = cv::abs(p);
  abs_p.setTo(1.0, abs_p > 1.0);
    
	cv::divide(cv::abs(xi), abs_p, temp);
  //If all of temp elements are lower than TOLX, algorithm terminates
	if ( cv::checkRange(temp, true, nullptr, 0.0, TOLX) ) return 1;   //minimum reached
    
	g.copyTo(dg);
	g = dfunc(p);
    
	den = cv::max(fret, 1.0);
	cv::multiply(cv::abs(g), abs_p / den, temp);
	if ( cv::checkRange(temp, true, nullptr, 0.0, gtol) ) return 1;   //minimum reached
    
	dg  = g - dg;
	hdg = hessin * dg;
    
	fac = fae = sumdg = sumxi = 0.0;
    
  fac = dg.dot(xi);
	fae = dg.dot(hdg);
	sumdg = dg.dot(dg);
	sumxi = xi.dot(xi);
    
	if (fac > std::sqrt(EPS * sumdg * sumxi)) 
  {
		fac = 1.0/fac;
		fad = 1.0/fae;
		dg = fac * xi - fad * hdg;   //Vector that makes BFGS different form DFP method
      
    hessin += fac * xi * xi.t() - fad * hdg * hdg.t() + fae * dg * dg.t();
 	}
    		
	xi = -hessin * g;
  return 0;   //minumum not found yet
}

template<class T>
double Minimization::linmin(cv::Mat_<double>& p, cv::Mat_<double>& xi, T &func)
{
  F1dim<T> f1dim(p, xi, func);
  
	bracket(0.0,1.0,f1dim);  //initial bounds conditions a=0, b=1
	xmin = brent(f1dim);
	
  
  xi = xi * xmin;
  p = p + xi;
  
	return fmin;
}



template<class T>
double Minimization::brent(T &func)
{
  //brent method suposses the minimum has been bracket before whithin points xa, xb, xc
  auto shft3 = [](double &a, double &b, double &c, const double d){a=b;	b=c; c=d;};
  auto sign = [](double a, double b) {return b >= 0.0 ? std::abs(a) : -std::abs(a);};
  
	const int ITMAX = 100;
	const double CGOLD = 0.3819660;
	const double ZEPS = std::numeric_limits<double>::epsilon() * 1.0e-3;
	double a, b, d = 0.0, etemp, fu, fv, fw, fx;
	double p, q, r, tol1, tol2, u, v, w, x, xm;
	double e = 0.0;
	
	a = (ax < cx ? ax : cx);
	b = (ax > cx ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = func(x);
	for (unsigned int iter = 0; iter < ITMAX; ++iter)
  {
		xm = 0.5 * (a+b);
		tol2 = 2.0 * (tol1 = tol * std::abs(x) + ZEPS);
		if (std::abs(x-xm) <= (tol2-0.5*(b-a)))
    {
			fmin = fx;
			return xmin = x;
		}
		if (std::abs(e) > tol1)
    {
			r = (x-w) * (fx-fv);
			q = (x-v) * (fx-fw);
			p = (x-v) * q - (x-w) * r;
			q = 2.0 * (q-r);
			if (q > 0.0) p = -p;
			q = std::abs(q);
			etemp = e;
			e = d;
			if (std::abs(p) >= std::abs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
      {
				d = CGOLD * (e=(x >= xm ? a-x : b-x));
      }
			else
      {
				d = p/q;
				u = x+d;
				if (u-a < tol2 || b-u < tol2)
        {
					d = sign(tol1,xm-x);
        }
			}
		}
    else
    {
			d = CGOLD*(e=(x >= xm ? a-x : b-x));
		}
	
		u = (std::abs(d) >= tol1 ? x+d : x + sign(tol1,d));
		fu = func(u);
		if (fu <= fx)
    {
			if (u >= x) a=x; else b=x;
			shft3(v,w,x,u);
			shft3(fv,fw,fx,fu);
		}
    else
    {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x)
      {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			}
      else if (fu <= fv || v == x || v == w)
      {
				v=u;
				fv=fu;
			}
		}
	}
	throw CustomException("Too many iterations in brent 1D minimization method.");
}



//member function definition have to be in header becaouse it's a template
template <class T>
void Minimization::bracket(const double& a, const double& b, T &func)
{
  auto shft3 = [](double &a, double &b, double &c, const double d){a=b;	b=c; c=d;};
  auto sign = [](double a, double b) {return b >= 0.0 ? std::abs(a) : -std::abs(a);};
  const double GOLD = 1.618034, GLIMIT = 100.0, TINY = 1.0e-20;
  ax=a; bx=b;
	double fu;
	fa = func(ax);
	fb = func(bx);
	if (fb > fa)
  {
		std::swap(ax, bx);
		std::swap(fb, fa);
	}
	cx = bx + GOLD * (bx-ax);
	fc = func(cx);
	while (fb > fc)
  {
		double r = (bx-ax) * (fb-fc);
		double q = (bx-cx) * (fb-fa);
		double u = bx - ((bx-cx)*q-(bx-ax)*r)/(2.0*sign(std::max(std::abs(q-r),TINY),q-r));
		double ulim = bx + GLIMIT * (cx-bx);
		if ((bx-u) * (u-cx) > 0.0)
    {
			fu = func(u);
			if (fu < fc)
      {
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				return;
			}
      else if (fu > fb) 
      {
				cx = u;
				fc = fu;
				return;
			}
			u = cx + GOLD * (cx-bx);
			fu = func(u);
		}
    else if ((cx-u) * (u-ulim) > 0.0) 
    {
			fu = func(u);
			if (fu < fc) 
      {
				shft3(bx, cx, u, u + GOLD * (u-cx));
				shft3(fb, fc, fu, func(u));
			}
		}
    else if ((u-ulim) * (ulim-cx) >= 0.0)
    {
			u = ulim;
			fu = func(u);
		}
    else 
    {
	  	u = cx + GOLD * (cx-bx);
			fu = func(u);
		}
		shft3(ax, bx, cx, u);
		shft3(fa, fb, fc, fu);
	}
}


#endif /* MINIMIZATION_H_ */
