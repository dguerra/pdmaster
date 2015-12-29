/*
 * Minimization.cpp
 *
 *  Created on: Sep 17, 2014
 *      Author: dailos
 */

#include <limits>  //numeric_limit<double>::epsilon
#include <algorithm>    // std::max
#include <cmath>   //std::abs
#include <functional>   //function objects
#include "Minimization.h"

Minimization::Minimization()
{
  // TODO Auto-generated constructor stub

}

Minimization::~Minimization()
{
  // TODO Auto-generated destructor stub
}

cv::Mat Minimization::gradient_diff(cv::Mat x, const std::function<double(cv::Mat)>& func)
{ //in case gradient function is not available, it could be build with a difference approximation
  //make up gradient vector through slopes and tiny differences
  double EPS(1.0e-8);
  cv::Mat df = cv::Mat::zeros(x.size(), x.type());
  cv::Mat xh = x.clone();
  double fold = func(x);
  for(unsigned int j = 0; j < x.total(); ++j)
  {
    double temp = x.at<double>(j,0);
    double h = EPS * std::abs(temp);
    if (h == 0) h = EPS;
    xh.at<double>(j,0) = temp + h;
    h = xh.at<double>(j,0) - temp;
    double fh = func(xh);
    xh.at<double>(j,0) = temp;
    df.at<double>(j,0) = (fh-fold)/h;
  }
  return df;
};


void Minimization::minimize(cv::Mat &p, const cv::Mat &Q2,
                            const std::function<double(cv::Mat)>& func, const std::function<cv::Mat(cv::Mat)>& dfunc, const Method& method)
{ 
  //Lambda function that turn minimize function + constraints problem into minimize function lower dimension problem
  auto F_constrained = [] (cv::Mat x, std::function<double(cv::Mat)> func, const cv::Mat& Q2) -> double
  {
    return func(Q2*x);
  };

  auto DF_constrained = [] (cv::Mat x, std::function<cv::Mat(cv::Mat)> dfunc, const cv::Mat& Q2) -> cv::Mat
  {
    return Q2.t() * dfunc(Q2*x);
  };
    
  std::function<double(cv::Mat)> f_constrained = std::bind(F_constrained, std::placeholders::_1, func, Q2);
  std::function<cv::Mat(cv::Mat)> df_constrained = std::bind(DF_constrained, std::placeholders::_1, dfunc, Q2);
  //Define a new starting point with lower dimensions after reduction with contraints
  cv::Mat p_constrained = Q2.t() * p;
  
  
  if(method == Method::BFGS)   //perform_BFGS
  {
    perform_BFGS(p_constrained, iter_, fret_, f_constrained, df_constrained);
  }
  else  if(method == Method::FISTA)   //perform_FISTA
  {
        //double lambda = 1.5;
    double lambda = 1.0;
    auto perform_soft_thresholding = [lambda](cv::Mat x, double tau)-> cv::Mat
    {
      //   Proximal operator for the scalar L1 norm.
      //   y = prox_{tau*|.|_1}(x) = max(0,1-tau/|x|)*x
      //y = max(0,1-tau./max(abs(x),1e-10)).*x;
      return cv::max(0.0, 1.0 - (lambda*tau)/cv::abs(x)).mul(x);
    };
    
    double L = 1000.0;
    //double L = 1.0;     //default
    auto L1_norm = [lambda](cv::Mat x) -> double
    {
      return lambda * norm(x, cv::NORM_L1);
    };
    
    //perform_fista(p, L1_norm, func, perform_soft_thresholding, dfunc, L);
    perform_FISTA(p_constrained, L1_norm, f_constrained, perform_soft_thresholding, df_constrained, L); 
  }
  
  p = Q2 * p_constrained;   //Go back to original dimensional 
}


void Minimization::perform_IHT( const cv::Mat& observation, const cv::Mat& measurement, cv::Mat&  x0, const unsigned int& sparsity, 
                                const double& mu)
{
   //Keep only k largest  coefficients
  auto hardThreshold = [](cv::Mat& p, const unsigned int& k)
  {
    cv::Mat mask = cv::Mat::ones(p.size(), CV_8U);
    cv::Mat pp(cv::abs(p));
    for(unsigned int i=0;i<k;++i)
    {
      cv::Point maxLoc;
      cv::minMaxLoc(pp, nullptr, nullptr, nullptr, &maxLoc, mask);
      mask.at<char>(maxLoc.y, maxLoc.x) = 0;
    }
    p.setTo(0.0, mask);
  };
  unsigned int numberOfIterations(200);
  for(unsigned int j =0; j<numberOfIterations; ++j)
  {
    //x0 = x0 + measurement.t() * (mu * (observation - (measurement*x0) ));
    
    cv::gemm(measurement, mu * (observation - (measurement*x0)), 1.0, x0, 1.0, x0, cv::GEMM_1_T);
    hardThreshold(x0, sparsity);
    std::cout << "x0: " << x0.t() << std::endl;
  }
}



//Perform FISTA method
void Minimization::perform_FISTA(cv::Mat& x, const std::function<double(cv::Mat)>& f, const std::function<double(cv::Mat)>& g, 
                               const std::function<cv::Mat(cv::Mat, double)>& ProxF, const std::function<cv::Mat(cv::Mat)>& GradG, double L)
{
//   Solves min_x g(x) + f(x)
//   e.g.   g(x) = |Ax-y|^2  &&  f(x) = L1(x)
//   where g is a smooth convex proper function and f is a
//   convex proper function with an easy to compute proximal operator.

//   http://www.mathworks.com/matlabcentral/fileexchange/16204-toolbox-sparse-optmization/content/toolbox_optim/perform_fb.m
//   http://www.mathworks.com/matlabcentral/fileexchange/16204-toolbox-sparse-optmization/content/toolbox_optim/tests/html/test_l1_lagrangian.html
//   https://github.com/gpeyre/matlab-toolboxes/blob/master/toolbox_optim/toolbox/perform_soft_thresholding.m
//   https://github.com/gpeyre/matlab-toolboxes/blob/master/toolbox_optim/perform_fb.m#L29
//   https://github.com/torch/optim/blob/master/fista.lua


  unsigned int maxline(20);
  unsigned int maxiter(1000);
  double Lstep(2.5);
  
  double t = 1.0;
  cv::Mat y = x.clone();
  cv::Mat x0 = x.clone();
  double fval =  std::numeric_limits<double>::max();
  double epsilon = 0.0000000001; //0.0000001; //std::numeric_limits<double>::epsilon();
    
  for (unsigned int i=0;i<maxiter;++i)
  {
    cv::Mat xnew;
    double gxnew;
    std::cout << "L: " << L << " ";
    
    for(unsigned int nline=0;nline<maxline;++nline)
    {
      cv::Mat GradGy = GradG(y).clone();
      xnew = ProxF( y - (1.0/L)*GradGy, (1.0/L) );
      cv::Mat stp = xnew-y;
      gxnew = g(xnew);
      if (gxnew <= g(y) + stp.dot(GradGy) + (L/2.0) * stp.dot(stp) + f(xnew) )  break;
      else L = L * Lstep;
      std::cout << L << " " << std::endl;
    }
    
    double fvalnew = gxnew + f(xnew);
    if( (fvalnew - fval) < epsilon && (fvalnew - fval) > -epsilon) 
    {
      std::cout << "at step " << i <<  ". minimum reached: " << fvalnew << std::endl;
      std::cout << "x = " << x.t() << std::endl;  
      break;
    }
    
    fval = fvalnew;
    //cv::Mat xnew = ProxF( y - (1.0/L)*GradG(y), (1.0/L) );    ///CAUTION
    double tnew = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
    y = xnew + (t - 1.0)/(tnew) * (xnew - x);
    x = xnew.clone();
    t = tnew;
    
    std::cout << "step " << i << " to minimum. " <<  "fret: " << fval << std::endl;
    std::cout << "x = " << x.t() << std::endl;
  }
}



//perform_BFGS
void Minimization::perform_BFGS(cv::Mat &p, int &iter, double &fret, 
                          std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc)
{
	int n = p.total();   //Check the vector has only one column first
  //Declare and initialize some variables: g = gradient, xi = direction, hessin = hessian matrix
	cv::Mat g, xi;
	cv::Mat hessin = cv::Mat::eye(n, n, cv::DataType<double>::type);  //initialize to identity matrix
  
	fret = func(p);
	g = dfunc(p);
  
  xi = -1 * g;  //first direction is the opposite to the gradient
  std::cout << "starting direction xi: " << xi.t() << std::endl;
  
  //variables: p:[point], xi:[search direction], func:[function], dfunc:[gradient function], g:[gradient at p], h:[hessian at p], 
  
	for (int its=0;its<ITMAX;its++)
  {
		iter = its;
    std::cout << "step " << iter << " to minimum. " << "fret: " << fret << std::endl;
    std::cout << "p = " << p.t() << std::endl;
    if(nextStep(p, xi, g, hessin, fret, func, dfunc)) return;   //minimum reached
	}
	throw("too many iterations in dfpmin");
}

int Minimization::nextStep(cv::Mat &p, cv::Mat &xi, cv::Mat &g, cv::Mat &hessin, double &fret, 
                           std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc)
{
  const double EPS = std::numeric_limits<double>::epsilon();
  //const double TOLX = 4 * EPS;
  const double TOLX = 3.0e-8;
  double den, fac, fad, fae, sumdg, sumxi;
  cv::Mat dg, hdg;
  //we use linmin uses brent method inside to look for the minimum in 1D
  fret = brentLineSearch(p, xi, func);
				
  cv::Mat temp;
  cv::Mat abs_p = cv::abs(p);
  abs_p.setTo(1.0, abs_p > 1.0);
    
  cv::divide(cv::abs(xi), abs_p, temp);
  //If all of temp elements are lower than TOLX, algorithm terminates
  if ( cv::checkRange(temp, true, nullptr, 0.0, TOLX) ){std::cout << "minimum reached" << std::endl; return 1; }   //minimum reached
  
  g.copyTo(dg);
  g = dfunc(p);
  
  den = cv::max(fret, 1.0);
  cv::multiply(cv::abs(g), abs_p / den, temp);
  
  if ( cv::checkRange(temp, true, nullptr, 0.0, gtol) ){std::cout << "minimum reached" << std::endl; return 1; }  //minimum reached
    
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

double Minimization::brentLineSearch(cv::Mat& p, cv::Mat& xi, std::function<double(cv::Mat)> &func)
{
  //Helpter function that turns a multidimensional functor into 1-dim, through point p and direction xi on function func
  auto F1dim = [] (const double &x, const cv::Mat &p, const cv::Mat &xi, const std::function<double(cv::Mat)> &func) -> double
  { //could be implemented through function adaptors, read more about it
    return func(p + x * xi);
  };
  
  std::function<double(double)> f1dim = std::bind(F1dim, std::placeholders::_1, p, xi, func);
  
  bracket(0.0,1.0,f1dim);  //initial bounds conditions a=0, b=1
  xmin = brent(f1dim);
	  
  xi = xi * xmin;
  p = p + xi;

  return fmin;
}


double Minimization::brent(std::function<double(double)> &func)
{
  //brent method suposses the minimum has been bracket before whithin points xa, xb, xc, member variables
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

void Minimization::bracket(const double& a, const double& b, std::function<double(double)> &func)
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
