/*
 * Constraints.cpp
 *
 *  Created on: Jan 10, 2015
 *      Author: dailos
 */

#include "Constraints.h"


Constraints::Constraints()
{
  
}

Constraints::~Constraints()
{
  // TODO Auto-generated destructor stub
}



void Constraints::householder(const cv::Mat &m, cv::Mat &R, cv::Mat &Q)
{
  auto matrix_minor = [](cv::Mat x, int d) -> cv::Mat
  { //For d = 2, turns this into this:
    // 5, 5, 5, 5, 5      1, 0, 0, 0, 0
    // 5, 5, 5, 5, 5      0, 1, 0, 0, 0
    // 5, 5, 5, 5, 5  ->  0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
  	cv::Mat m = cv::Mat::zeros(x.size(), x.type());
  	for (int i = 0; i < d; i++)
    {
		  m.at<double>(i,i) = 1;
    }
  	for (int i = d; i < x.rows; i++)
    {
		  for (int j = d; j < x.cols; j++)
      {
	  		m.at<double>(i,j) = x.at<double>(i,j);
      }
    }
  
	  return m;
  };

  
  std::vector<cv::Mat> q;
	cv::Mat z = m.clone(), z1;
  
	for (int k = 0; k < m.cols && k < m.rows - 1; k++) 
  {
		double a;
    cv::Mat  x, e = cv::Mat::zeros(m.rows, 1, m.type());
    
		z1 = matrix_minor(z, k);
		
		z = z1.clone();
    z.col(k).copyTo(x);
		a = std::sqrt(x.dot(x));
    
		//if (m.at<double>(k,k) > 0) a = -a;   //It changes some signs!!
 
    e.at<double>(k,0) = 1.0;
 
    e = x + e * a;
		e = e / std::sqrt(e.dot(e));
    
		q.push_back( cv::Mat::eye(e.rows, e.rows, e.type()) - 2.0 * e * e.t() );
		z1 = q.back() * z;
		
		z = z1.clone();
	}

	q.front().copyTo(Q);
	R = q.front() * m;
	for (int i = 1; i < m.cols && i < m.rows - 1; i++) 
  {
		z1 = q.at(i) * Q;
		Q = z1.clone();
	}
  
	z = Q * m;
	
	R = z.clone();
	Q = Q.t();
}
