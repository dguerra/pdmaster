/*
 * Constraints.h
 *
 *  Created on: Jan 10, 2015
 *      Author: dailos
 */

#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

class Constraints
{
public:
  Constraints();
  virtual ~Constraints();
  void householder(const cv::Mat &m, cv::Mat &R, cv::Mat &Q);  //QR decompose matrix m into Q*R (where Q orthogonal and with null space)
  enum class PDConfiguration {KnownDifferencesInPhase, PartlyKnownDifferencesInPhase, Speckle, ShackHartmann}; 

private:
  cv::Mat Q2_;
  
};

#endif /* CONSTRAINTS_H_ */
