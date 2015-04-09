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

//ANEXO:
  //in case gradient function is not available, it could be build with a difference approximation
  //    std::function<cv::Mat(cv::Mat)> dfuncc_diff = [objFunction] (cv::Mat x) -> cv::Mat
//    { //make up gradient vector through slopes and tiny differences
//      double EPS(1.0e-8);
//      cv::Mat df = cv::Mat::zeros(x.size(), x.type());
//  	  cv::Mat xh = x.clone();
//  	  double fold = objFunction(x);
//      for(unsigned int j = 0; j < x.total(); ++j)
//      {
//        double temp = x.at<double>(j,0);
//        double h = EPS * std::abs(temp);
//        if (h == 0) h = EPS;
//        xh.at<double>(j,0) = temp + h;
//        h = xh.at<double>(j,0) - temp;
//        double fh = objFunction(xh);
//        xh.at<double>(j,0) = temp;
//        df.at<double>(j,0) = (fh-fold)/h;    
//      }
//      return df;  
//    };
    

Minimization::Minimization()
{
  // TODO Auto-generated constructor stub

}

Minimization::~Minimization()
{
  // TODO Auto-generated destructor stub
}

cv::Mat Minimization::gradient_diff(cv::Mat x, const std::function<double(cv::Mat)>& func)
{ //make up gradient vector through slopes and tiny differences
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

//const std::function<cv::Mat(cv::Mat, cv::Mat)>& sup_gp   -> nonsmooth case
void Minimization::minimize(cv::Mat &p, const cv::Mat &Q2,
                            const std::function<double(cv::Mat)>& func, const std::function<cv::Mat(cv::Mat)>& dfunc)
{ 
  //Lambda function that turn minimize function + constraints problem into minimize function lower dimension problem
  auto F_constrained = [] (cv::Mat x, std::function<double(cv::Mat)> func, const cv::Mat& Q2) -> double
  {
    return func(Q2*x);
  };

  auto DF_constrained = [] (cv::Mat x, std::function<cv::Mat(cv::Mat)> dfunc, const cv::Mat& Q2) -> cv::Mat
  {
    cv::Mat subdiff_constrained;
    cv::Mat subdiff(dfunc(Q2*x));
    if(subdiff.channels() == 2)
    {
      cv::Mat sd_bounds[2];   //Split subdifferential set into its bounds (range begin and end of each dimension)
      cv::split(subdiff, sd_bounds);
      cv::Mat bounds_constrained[2];
      bounds_constrained[0] = Q2.t() * sd_bounds[0];
      bounds_constrained[1] = Q2.t() * sd_bounds[1];
      cv::merge(bounds_constrained, 2, subdiff_constrained);
    }
    else if(subdiff.channels() == 1)
    {
      subdiff_constrained = Q2.t() * subdiff;
    }
    else throw CustomException("Minimization ERROR: Wrong subdifferential format.");
    return subdiff_constrained;
  };
    
  std::function<double(cv::Mat)> f_constrained = std::bind(F_constrained, std::placeholders::_1, func, Q2);
  std::function<cv::Mat(cv::Mat)> df_constrained = std::bind(DF_constrained, std::placeholders::_1, dfunc, Q2);
  //Define a new starting point with lower dimensions after reduction with contraints
  cv::Mat p_constrained = Q2.t() * p;
  dfpmin(p_constrained, iter_, fret_, f_constrained, df_constrained);
  p = Q2 * p_constrained;   //Go back to original dimensional 
}

int Minimization::descentDirection(const cv::Mat& subdifferential, const cv::Mat& Bt, const std::function<cv::Mat(cv::Mat, cv::Mat)>& sup_gp, cv::Mat& p, cv::Mat& gnew)
{
  int success = 1;
  if(subdifferential.channels() == 1)  //smooth case: only one subgradient value for each dimension
  {
    p = -Bt * subdifferential;
    subdifferential.copyTo(gnew);
    return success;
  }
  
  cv::Mat sd_bounds[2];   //Split subdifferential set into its bounds (range begin and end of each dimension)
  cv::split(subdifferential, sd_bounds);
  
  if( cv::checkRange(sd_bounds[0]-sd_bounds[1], true, nullptr, 0.0, 0.0) )  //diferentiable point within nonsmooth function
  {
    p = -Bt * sd_bounds[0];
    sd_bounds[0].copyTo(gnew);
    return success;
  }

  //GetDescentDirection(TheMatrix &g, const vector<int> &hinges, vector<double> &hingebetas, TheMatrix & gnew, TheMatrix & p)
	double gp = -1;
	double gap = 0.0;
	double eps = 0.0;
	double min_M = 0.0;
	double gp_old = 0.0;
	double gBg = 0.0;
	int trynum = 0;
	double best_gp = -1;

  cv::Mat g0(sd_bounds[0]);     //First subgradient: random from subdifferencial set, lowest in this case

	cv::Mat ga(g0);
  double epsilonTol = 3.0e-8;

	g0.copyTo(gnew);
	gnew.copyTo(p);

	//BMult(p);
	//p.Scale(-1.0);
	p = -Bt * g0;
	cv::Mat bestDir(p);

  gp_old = gnew.dot(p);
  gnew = sup_gp(subdifferential, p);
  gp = gnew.dot(p);
	best_gp = gp;

	//note initial <ga, p> = <g1, p1> =: gp_old
	gap = gp_old;
	min_M = gp - 0.5 * gap;
	eps = min_M - 0.5 * gap;

	while ((gp > 0.0 || eps > epsilonTol) && trynum < ITMAX) 
  {
		double numerator;
		double denominator;
		double mu;

		numerator = gp - gap;
		cv::Mat Bg(Bt * gnew);
		// <Bg^{i+1}>
		// calculate <g^{i+1}, Bg^{i+1}>
		gBg = gnew.dot(Bg);
		// denominator = <g^{i+1}, Bg^{i+1}> + 2 <g^{i+1}, p^{i}> - <ga^{i}, p^{i}>
		denominator = gBg + 2 * gp - gap;
		mu = std::min(1.0, numerator / denominator);
		if (mu <= 0) 
    {
			std::cout << "encountered mu = " << mu << std::endl;
			break;
		}

    ga = ((1.0 - mu) * ga) + mu * gnew;  //Updates ga
		p = (1.0 - mu) * p - mu * Bg;   //Updates direction p

		//<ga^{i+1},p^{i+1}> = (1-mu)^2 <ga^{i}, p^{i}> + mu(1-mu)<g^{i+1}, p^{i}> + mu<g^{i+1}, p^{i+1}>
		gap = (1 - mu) * (1 - mu) * gap + (1 - mu) * mu * gp;
    
    gp_old = gnew.dot(p);
		gnew = sup_gp(subdifferential, p);
    gp = gnew.dot(p);

		//calculate <ga, p>=ga.Dot(p, gap) in a more efficient way
		gap += mu * gp_old;

		double current_M = gp - 0.5 * gap;
		if (current_M < min_M) {
			min_M = current_M;
			p.copyTo(bestDir);
			best_gp = gp;
		}

		// bound on Duality Gap
		eps = min_M - 0.5 * gap;
		trynum++;

   	std::cout << "mu: " << mu << "gp: " << gp << "eps: " << eps << "minM: " << min_M << std::endl;
	}

	if (trynum > 0) {
		bestDir.copyTo(p);
	}

	// total dir findings: dirFinding += trynum;
	if (best_gp > 0) {
	 	success = 0;
	}
	return success;
}

void Minimization::dfpmin(cv::Mat &p, int &iter, double &fret, 
                          std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc)
{
	int n = p.total();   //Check the vector has only one column first
  //Declare and initialize some variables: g = gradient, xi = direction, hessin = hessian matrix
	cv::Mat g, xi, subdifferential;
	cv::Mat hessin = cv::Mat::eye(n, n, cv::DataType<double>::type);  //initialize to identity matrix
  
	fret = func(p);
	subdifferential = dfunc(p);
  
  //sup_gp function for testing
  std::function<cv::Mat(cv::Mat, cv::Mat)> sup_gp = [] (cv::Mat subdiff, cv::Mat dir) -> cv::Mat
  {
    if(subdiff.channels() == 1) return subdiff;
      
    cv::Mat sd_bounds[2];   //Split subdifferential set into its bounds (range begin and end of each dimension)
    cv::split(subdiff, sd_bounds);
    cv::Mat g = cv::Mat::zeros(dir.size(), dir.type());
    cv::add(g, sd_bounds[0], g, dir>0);
    cv::add(g, sd_bounds[1], g, dir<=0);
    return g;
  };
  
  //variables: p:[point], xi:[search direction], func:[function], dfunc:[gradient function], g:[gradient at p], h:[hessian at p], 
 	const double EPS = std::numeric_limits<double>::epsilon();
	//const double TOLX = 4 * EPS;   //also std::sqrt(EPS)
	const double TOLX = 3.0e-8;
  
	for (int its=0;its<ITMAX;its++)
  {
    iter = its;
    descentDirection(subdifferential, hessin, sup_gp, xi, g);
    std::cout << "step " << iter << " to minimum. " << "fret: " << fret << std::endl << "p = " << p.t() << std::endl;

    double den, fac, fad, fae, sumdg, sumxi;
    cv::Mat dg, hdg;
    //we use linmin uses brent method inside to look for the minimum in 1D
    //fret = brentLineSearch(p, xi, func);
    fret = armijoWolfeLineSearch(p, xi, func, dfunc);
    cv::Mat temp;
	  cv::Mat abs_p = cv::abs(p);
    abs_p.setTo(1.0, abs_p > 1.0);

	  cv::divide(cv::abs(xi), abs_p, temp);
    //If all of temp elements are lower than TOLX, algorithm terminates
    if ( cv::checkRange(temp, true, nullptr, 0.0, TOLX) ){std::cout << "minimum reached" << std::endl; return; }   //minimum reached

	  g.copyTo(dg);
    subdifferential = dfunc(p);
    
    if(subdifferential.channels() == 2)
    {
      cv::Mat sd_bounds[2];   //Split subdifferential set into its bounds (range begin and end of each dimension)
      cv::split(subdifferential, sd_bounds);
      sd_bounds[1].copyTo(g);   //Select random subgradient from subdifferential set
    }
    else
    {
      subdifferential.copyTo(g);
    }
    //Choose g, subgradient from subdifferential set such that: xi.t() * (g - dg) > 0;
	  if( xi.dot(g-dg) <= 0) throw CustomException("FATAL ERROR: Secant condition not verified. Look further into this.");

	  den = cv::max(fret, 1.0);
	  cv::multiply(cv::abs(g), abs_p / den, temp);
    
    if( cv::checkRange(temp, true, nullptr, 0.0, gtol) ){std::cout << "minimum reached" << std::endl; return; }  //minimum reached
    
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
      
      hessin += fac * xi * xi.t() - fad * hdg * hdg.t() + fae * dg * dg.t();   //Updates inverse hessian
 	  }
	}
	throw("too many iterations in dfpmin");
}

/*
int Minimization::nextStep(cv::Mat &p, cv::Mat &xi, cv::Mat &g, cv::Mat &hessin, double &fret, 
                           std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc)
{
 
	const double EPS = std::numeric_limits<double>::epsilon();
	//const double TOLX = 4 * EPS;   //also std::sqrt(EPS)
	const double TOLX = 3.0e-8;
  double den, fac, fad, fae, sumdg, sumxi;
  cv::Mat dg, hdg;
  //we use linmin uses brent method inside to look for the minimum in 1D
  //fret = brentLineSearch(p, xi, func);
  fret = armijoWolfeLineSearch(p, xi, func, dfunc);

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
*/
double Minimization::armijoWolfeLineSearch(cv::Mat& p, cv::Mat& xi, std::function<double(cv::Mat)> &func, std::function<cv::Mat(cv::Mat)> &dfunc)
{ 
  //Helpter function that turns a multidimensional functor into 1-dim, through point p and direction xi on function func
  auto F1dim = [] (const double &x, const cv::Mat &p, const cv::Mat &xi, const std::function<double(cv::Mat)> &func) -> double
  { //could be implemented through function adaptors, read more about it
    return func(p + x * xi);
  };
  //gradient from point p, through line xi, projected onto direction search xi
  auto Df1dim = [] (const double &x, const cv::Mat &p, const cv::Mat &xi, std::function<cv::Mat(cv::Mat)> &dfunc) -> double
  { //could be implemented through function adaptors, read more about it
    cv::Mat subdiff1dim = dfunc(p + x * xi);
    cv::Mat g1d;
    if(subdiff1dim.channels() == 2)
    {
      cv::Mat sd_bounds1d[2];   //Split subdifferential set into its bounds (range begin and end of each dimension)
      cv::split(subdiff1dim, sd_bounds1d);
      sd_bounds1d[0].copyTo(g1d);
    }
    else if(subdiff1dim.channels() == 1)
    {
      subdiff1dim.copyTo(g1d);
    }
    return g1d.dot(xi);
  };
  
  std::function<double(double)> f1dim = std::bind(F1dim, std::placeholders::_1, p, xi, func);
  std::function<double(double)> df1dim = std::bind(Df1dim, std::placeholders::_1, p, xi, dfunc);
  
  bool done(false);
  double C1 = 1e-4, C2 = 0.9;    //orignal configuration
  //double C1 = 1e-7, C2 = 0.99999;
  
  double alpha = 0.0;		/* lower bound on step length*/
  double beta_max = 1e100;	/* upper bound on step length*/
  double ftarget = 0.0;
  double beta  = beta_max;	
    
  double t = 1.0;               /* try step length 1 first*/
  int nbisect(0), nexpand(0);
  double dnorm = cv::norm(xi, cv::NORM_L2);
  
  int nbisectmax(30), nexpandmax(30);
  if (0.0 == dnorm)
  {
    nbisectmax = 100;
    nexpandmax = 1000;
  }
  else
  {
    nbisectmax = std::ceil( std::log( 100000 * dnorm )/std::log(2.0) );
    if (nbisectmax < 100) nbisectmax = 100;
    nexpandmax = std::ceil( std::log( 100000 / dnorm )/std::log(2.0) );    
    if (nexpandmax < 100) nexpandmax = 100;
  }

  double f0 = f1dim(0.0);
  double g0 = df1dim(0.0);     //first gradient porjected onto search direction xi
    
  double armijo_rhs_p = g0 * C1;
  double wwolfe_rhs   = g0 * C2;
  double f(0.0), g(0.0);
  while (!done)
  {
    f = f1dim(t);
    g = df1dim(t);
    //if (f < ftarget) {done = true; break;}
    
    if (f > f0 + (t * armijo_rhs_p) ) beta = t;    //armijo fails, gone too far
    else if (g < wwolfe_rhs) alpha = t;    //weak wolfe fails, not gone far enough
    else { alpha = t; beta = t; done = true;}   //both conditions are ok, so quit
    
    if (beta < beta_max) 
    {
      if (nbisect < nbisectmax) {t = (alpha + beta) / 2; nbisect++;}
      else done = true;
    }
    else
    {
      if (nexpand < nexpandmax) {t = 2 * alpha; nexpand++;}
      else done = true;  //Expansion bigger than expandmax
    }
  }

  xi = xi * t;
  p = p + xi;

  return f1dim(t);
  //return f;
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


